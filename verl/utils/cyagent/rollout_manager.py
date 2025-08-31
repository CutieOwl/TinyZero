import fcntl
import os
import time
import torch

from datetime import datetime, timezone
from tensordict import TensorDict

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

class RolloutManager:
    def __init__(self, max_iterations, actor_rollout_wg, tokenizer, log_dir, max_prompt_length, truncation):
        self.max_iterations = max_iterations
        self.actor_rollout_wg = actor_rollout_wg
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir
        # self.verl_dir = os.path.join(self.log_dir, 'verl')
        # self.cybench_dir = os.path.join(self.log_dir, 'cybench')
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def add_response_mask_to_output(self, output, response_idxs, prompt_start_idx, output_idx, orig_rollout_ids):
        # create tensor in same shape of attention mask
        start_time = datetime.now(timezone.utc)
        response_mask = torch.zeros_like(output.batch['attention_mask'])
        prompt_ids = output.batch['prompts'][output_idx]
        response_ids = output.batch['responses'][output_idx]
        prompt_and_response_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        
        # agent_header = "----------Message from agent----------\n"
        # Don't use agent header actually
        # agent_header_length = 5 # len(self.tokenizer.encode(agent_header, add_special_tokens=False))
        # print(f"Agent header length in tokens: {agent_header_length}")

        # set the response part of the attention mask to 1
        for i, (start, end) in enumerate(response_idxs):
            real_start = start + prompt_start_idx
            real_end = end + prompt_start_idx
            response_mask[:, real_start:real_end] = 1
            # Use the below to make sure the response mask is correct
            # response_str = self.tokenizer.decode(prompt_and_response_ids[real_start:real_end])
            # print(f"Response string start {real_start} end {real_end}: {response_str}")
    
        # add the response mask to the output
        output.batch['response_mask'] = response_mask
        # also add the response_idxs to the output
        # output.non_tensor_batch['response_idxs'] = response_idxs
        # also add the prompt_start_idx to the output
        # output.non_tensor_batch['prompt_start_idx'] = prompt_start_idx
        output.non_tensor_batch['rollout_id'] = orig_rollout_ids

        # print(f"Response mask shape: {response_mask.shape}")
        # print(f"Response mask: {response_mask}")
        print(f"Response idxs: {response_idxs}")
        print(f"Prompt start idx: {prompt_start_idx}")
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        print(f"Time taken to add response mask: {duration} seconds")
        return output


    def get_single_rollout(self, gen_batch, output_idx):
        orig_rollout_ids = gen_batch.non_tensor_batch['rollout_id']
        rollout_id = orig_rollout_ids[output_idx]
        print(gen_batch.non_tensor_batch["rollout_id"])
        response_idxs = []
        prompt_start_idx = 0
        for i in range(self.max_iterations):
            # print(f"Iteration {i + 1} gen_batch: {gen_batch},\n attn_mask before {gen_batch.batch['attention_mask'][output_idx]}")
            start_time = datetime.now(timezone.utc)
            output = self.actor_rollout_wg.generate_sequences(gen_batch)
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} time taken to generate: {duration}") # output: {output},\n , attn_mask after {output.batch['attention_mask'][output_idx]}
            # Decode the output and write to file
            start_time = datetime.now(timezone.utc)
            prompt_ids = output.batch['prompts'][output_idx]
            prompt_length = prompt_ids.shape[-1]
            unpadded_prompt_length = output.batch['attention_mask'][output_idx][:prompt_length].sum()
            prompt_start_idx = self.max_prompt_length - unpadded_prompt_length
            # print(f"Prompt length: {prompt_length}, unpadded prompt length: {unpadded_prompt_length}, prompt_start_idx: {prompt_start_idx}")

            response_ids = output.batch['responses'][output_idx]
            response_length = output.batch['attention_mask'][output_idx][prompt_length:].sum()
            # print(f"Response length: {response_length}")
            response_ids = response_ids[:response_length]

            cur_response_start = unpadded_prompt_length
            cur_response_end = unpadded_prompt_length + response_length
            response_idxs.append((cur_response_start, cur_response_end))
            # print(f"Cur response start: {cur_response_start}, cur response end: {cur_response_end}")
            response_str = self.tokenizer.decode(response_ids)

            with open(os.path.join(self.log_dir, f'{rollout_id}_{i}.out'), 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX) 
                f.write(response_str)
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f, fcntl.LOCK_UN) 

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} Time taken to decode response and write response to file {os.path.join(self.log_dir, f'{rollout_id}_{i}.out')}: {duration}")

            # Wait until we either receive a score or an in file
            start_time = datetime.now(timezone.utc)
            backoff = 0.01
            max_backoff = 0.1
            score_path = os.path.join(self.log_dir, f'{rollout_id}_{i}_score.txt')
            in_path = os.path.join(self.log_dir, f'{rollout_id}_{i+1}.in')
            print(f"Waiting for score file {score_path} or input file {in_path} to appear starting at {start_time}")
            done = False
            while True:
                if os.path.exists(score_path):
                    new_output = self.add_response_mask_to_output(
                        output,
                        response_idxs,
                        prompt_start_idx, 
                        output_idx, 
                        orig_rollout_ids
                    )
                    return new_output
                elif os.path.exists(in_path):
                    break
                else:
                    # print(f"Wokeup at {datetime.now(timezone.utc)}, did not find {score_path} or {in_path}")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} Time taken to check for score or input file: {duration}, end_time {end_time}")

            # If we reach here, it means we have a new input file to process
            with open(in_path, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH) 
                prompt_with_chat_template = f.read()
                fcntl.flock(f, fcntl.LOCK_UN)
            
            start_time = datetime.now(timezone.utc)
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation
            )
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            position_ids = compute_position_id_with_mask(attention_mask)
            position_ids = position_ids.to(self.device)

            # print(f"Input ids shape {input_ids.shape}")
            # print(f"attention mask shape {attention_mask.shape}")
            # print(f"position ids shape {position_ids.shape}")

            # gen_batch needs to have size multiple of world size because it will be split
            # across the gpus. Therefore for now I just repeat it across the world size
            new_gen_batch = TensorDict({
            'input_ids': input_ids.expand(self.actor_rollout_wg.world_size, -1),
            'attention_mask': attention_mask.expand(self.actor_rollout_wg.world_size, -1),
            'position_ids': position_ids.expand(self.actor_rollout_wg.world_size, -1),
            }, batch_size=self.actor_rollout_wg.world_size)
            gen_batch = DataProto(batch=new_gen_batch)
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} Time taken to tokenize and prepare new gen_batch: {duration}")

        # We should never get here, but if we do, return output
        print(f"SOMETHING WENT WRONG, reached max iterations {self.max_iterations} without receiving a score or an input file.")
        new_output = self.add_response_mask_to_output(
            output,
            response_idxs,
            prompt_start_idx,
            output_idx,
            orig_rollout_ids
        )
        return new_output

    def get_rollout(self, gen_batch):
        # print(f"Gen batch inside rollout manager: {gen_batch}")
        # Generate until we either reach the max iterations or receive an answer/score
        outputs = []
        torch.set_printoptions(threshold=float('inf'))
        batch_size = gen_batch.batch.batch_size[0] // self.actor_rollout_wg.world_size
        print(f"Batch size: {batch_size}")
        chunks = gen_batch.chunk(batch_size) 
        for chunk in chunks:
            print(f"Processing chunk {chunk}")
            for output_idx in range(chunk.batch.batch_size[0]):
                repeated_output = self.get_single_rollout(chunk, output_idx)
                # print(f"Got repeated output {repeated_output}")
                repeated_chunks = repeated_output.chunk(chunk.batch.batch_size[0])
                # print(f"Got repeated chunks {repeated_chunks}")
                output = repeated_chunks[output_idx]
                # print(f"Got output {output}")
                # print(f"Output attn mask {output.batch['attention_mask']}")
                outputs.append(output)

        # print(f"Outputs {outputs}")

        # Concatenate all the responses, prompts, input_ids, attention_masks, and position_ids
        start_time = datetime.now(timezone.utc)
        ret_output = DataProto.concat(outputs)
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        print(f"Time taken to concat: {duration} seconds")
        # print(f"output response mask {ret_output.batch['response_mask']}")
        return ret_output


                

            
            
        

