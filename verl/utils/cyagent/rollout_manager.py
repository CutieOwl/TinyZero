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
        self.verl_dir = os.path.join(self.log_dir, 'verl')
        self.cybench_dir = os.path.join(self.log_dir, 'cybench')
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def get_single_rollout(self, gen_batch, rollout_id, output_idx):
        print(gen_batch.non_tensor_batch["rollout_id"])
        for i in range(self.max_iterations):
            print(f"Iteration {i + 1} gen_batch: {gen_batch},\n attn_mask before {gen_batch.batch['attention_mask'][output_idx]}")
            start_time = datetime.now(timezone.utc)
            output = self.actor_rollout_wg.generate_sequences(gen_batch)
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} output: {output},\n time taken to generate: {duration}, attn_mask after {gen_batch.batch['attention_mask'][output_idx]}")
            # Decode the output and write to file
            start_time = datetime.now(timezone.utc)
            prompt_ids = output.batch['prompts'][output_idx]
            prompt_length = prompt_ids.shape[-1]

            response_ids = output.batch['responses'][output_idx]
            response_length = output.batch['attention_mask'][output_idx][prompt_length:].sum()
            response_ids = response_ids[:response_length]
            response_str = self.tokenizer.decode(response_ids)

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} Time taken to decode response: {duration}")
            start_time = datetime.now(timezone.utc)

            with open(os.path.join(self.verl_dir, f'{rollout_id}_{i}.out'), 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX) 
                f.write(response_str)
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN) 

            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} Time taken to write response to file: {duration}")

            # Wait until we either receive a score or an in file
            start_time = datetime.now(timezone.utc)
            backoff = 0.01
            max_backoff = 0.1
            score_path = os.path.join(self.cybench_dir, f'{rollout_id}_{i}_score.txt')
            in_path = os.path.join(self.cybench_dir, f'{rollout_id}_{i+1}.in')
            print(f"Waiting for score file {score_path} or input file {in_path} to appear starting at {start_time}")
            done = False
            while True:
                if os.path.exists(score_path):
                    return output
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
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            print(f"Iteration {i + 1} Time taken to tokenize new prompt: {duration}")


            start_time = datetime.now(timezone.utc)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            position_ids = compute_position_id_with_mask(attention_mask)
            position_ids = position_ids.to(self.device)

            print(f"Input ids shape {input_ids.shape}")
            print(f"attention mask shape {attention_mask.shape}")
            print(f"position ids shape {position_ids.shape}")

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
            print(f"Iteration {i + 1} Time taken to prepare new gen_batch: {duration}")

        # We should never get here, but if we do, return output
        return output

    def get_rollout(self, gen_batch):
        print(f"Gen batch inside rollout manager: {gen_batch}")
        # Generate until we either reach the max iterations or receive an answer/score
        outputs = []
        torch.set_printoptions(threshold=float('inf'))
        batch_size = gen_batch.batch.batch_size[0] // self.actor_rollout_wg.world_size
        print(f"Batch size: {batch_size}")
        chunks = gen_batch.chunk(batch_size) 
        for chunk in chunks:
            print(f"Processing chunk {chunk}")
            for output_idx in range(chunk.batch.batch_size[0]):
                rollout_id = chunk.non_tensor_batch['rollout_id'][output_idx]
                repeated_output = self.get_single_rollout(chunk, rollout_id, output_idx)
                output = repeated_output[output_idx]
                print(f"Got output {output}")
                print(f"Output attn mask {output.batch['attention_mask']}")
                outputs.append(output)

        # Concatenate all the responses, prompts, input_ids, attention_masks, and position_ids
        print(f"Starting concat at {datetime.now(timezone.utc)}")
        ret_output = DataProto.concat(outputs)
        print(f"Finished concat at {datetime.now(timezone.utc)}")
        return ret_output


                

            
            
        

