import torch

class RolloutManager:
    def __init__(self, dataset, rollout_worker, cli_callback, num_steps=2, device=None):
        """
        Args:
            dataset: An instance of RLHFDataset or similar.
            rollout_worker: An instance of vLLMRollout or similar.
            cli_callback: A function that takes a response and returns CLI feedback.
            num_steps: Number of generate/CLI steps per rollout.
            device: torch.device or None.
        """
        self.dataset = dataset
        self.rollout_worker = rollout_worker
        self.cli_callback = cli_callback
        self.num_steps = num_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_rollout(self, idx):
        """
        Run a multi-step rollout for a single prompt.
        Returns the full trajectory (list of dicts or a custom structure).
        """
        row = self.dataset[idx]
        conversation = [row['input_ids'].unsqueeze(0).to(self.device)]
        meta_info = {'eos_token_id': self.dataset.tokenizer.eos_token_id}

        for step in range(self.num_steps):
            # Prepare DataProto or similar for vLLMRollout
            from verl import DataProto
            batch = {
                'input_ids': conversation[-1],
                'attention_mask': row['attention_mask'].unsqueeze(0).to(self.device),
                'position_ids': row['position_ids'].unsqueeze(0).to(self.device),
            }
            data_proto = DataProto(batch=batch, meta_info=meta_info)

            # Generate continuation
            result = self.rollout_worker.generate_sequences(data_proto)
            response = result.batch['responses']  # (1, seq_len)

            # Get CLI feedback
            cli_feedback = self.cli_callback(response)

            # Append response and feedback for next step
            conversation.append(response)
            # Optionally, encode CLI feedback as tokens and append
            # feedback_ids = self.dataset.tokenizer(cli_feedback, return_tensors='pt').input_ids.to(self.device)
            # conversation.append(feedback_ids)

        # After all steps, assemble the full trajectory
        return conversation

    def run_all(self, max_rollouts=None):
        """
        Run rollouts for the whole dataset (or up to max_rollouts).
        Returns a list of trajectories.
        """
        results = []
        n = len(self.dataset) if max_rollouts is None else min(max_rollouts, len(self.dataset))
        for idx in range(n):
            trajectory = self.run_rollout(idx)
            results.append(trajectory)
        return results