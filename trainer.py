import copy
import time
from torch.optim.lr_scheduler import LambdaLR

import numpy
import ray
import torch

import models
import networks.muzero_network as mz_net


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = mz_net.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = self.update_weights(batch)

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )

            if self.config.save_model and self.training_step % self.config.save_interval == 0:
                shared_storage.save_checkpoint.remote(self.training_step)
                shared_storage.save_buffer.remote(replay_buffer, self.training_step,
                    shared_storage.get_info.remote("num_played_games"),
                    shared_storage.get_info.remote("num_reanalysed_games"))


            if self.config.network == "double":
                value_loss, trans_value_loss = value_loss
                reward_loss, trans_reward_loss = reward_loss
                policy_loss, trans_policy_loss = policy_loss

                shared_storage.set_info.remote(
                    {
                        "training_step": self.training_step,
                        "trans_value_loss": trans_value_loss,
                        "trans_reward_loss": trans_reward_loss,
                        "trans_policy_loss": trans_policy_loss
                    })


            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def loss_loop_trans(self, predictions, targets, target_value_scalar, priorities, gradient_scale_batch, action_mask):
        (values, rewards, policy_logits) = predictions
        (target_value, target_reward, target_policy) = targets

        # Compute losses
        value_losses, reward_losses, policy_losses = self.loss_function(
            values, rewards, policy_logits, target_value, target_reward, target_policy
        )
        reward_losses[:,0] = 0.0
        # todo also no losses for the steps after the game concluded

        non_padding_mask = ~action_mask
        # Apply mask to losses
        value_losses = value_losses * non_padding_mask
        reward_losses = reward_losses * non_padding_mask
        policy_losses = policy_losses * non_padding_mask

        # Directly scale the losses, add an epsilon to avoid dividing by zero
        scaling_factors = gradient_scale_batch + 1e-8  # Shape: (batch_size, sequence_length)

        # Apply scaling from index 1 onwards
        value_losses[:, 1:] = value_losses[:, 1:] / scaling_factors[:, 1:]
        reward_losses[:, 1:] = reward_losses[:, 1:] / scaling_factors[:, 1:]
        policy_losses[:, 1:] = policy_losses[:, 1:] / scaling_factors[:, 1:]

        value_loss = value_losses.sum(dim=1)  # Shape: (batch_size,)
        reward_loss = reward_losses.sum(dim=1)
        policy_loss = policy_losses.sum(dim=1)

        if priorities is not None:
            B, T, D = values.shape
            merged_values = values.reshape(B * T, D)

            # todo consider going away from CPU
            pred_value_scalars = models.support_to_scalar(
                merged_values, self.config.support_size
            ).detach().cpu().numpy()
            pred_value_scalars = pred_value_scalars.reshape(B, T)
            priorities = (
                    numpy.abs(pred_value_scalars - target_value_scalar) ** self.config.PER_alpha
            )

        return value_loss, reward_loss, policy_loss, priorities




    def loss_loop_fast(self, predictions, targets, target_value_scalar, priorities, gradient_scale_batch):

        (target_value, target_reward, target_policy) = targets

        if self.config.network == "transformer":
            (values, rewards, policy_logits) = predictions
        else: #if True:
            values, rewards, policy_logits = zip(*predictions)  # Unpack predictions
            values = torch.stack(values, dim=1).squeeze(-1)  # Shape: (time_steps, batch_size)
            rewards = torch.stack(rewards, dim=1).squeeze(-1)  # Shape: (time_steps, batch_size)
            policy_logits = torch.stack(policy_logits, dim=1)  # Shape: (time_steps, batch_size, policy_size)

        # Compute losses
        value_losses, reward_losses, policy_losses = self.loss_function(
            values, rewards, policy_logits, target_value, target_reward, target_policy
        )

        reward_losses[:,0] = torch.zeros(reward_losses[:,0].shape, device=reward_losses.device)

        for i in range(1, value_losses.shape[-1]):
            value_losses[:, i].register_hook(lambda grad: grad / gradient_scale_batch[:, i])
            reward_losses[:, i].register_hook(lambda grad: grad / gradient_scale_batch[:, i])
            policy_losses[:, i].register_hook(lambda grad: grad / gradient_scale_batch[:, i])

        # this
        # scaled_value_losses = value_losses / gradient_scale_batch.unsqueeze(0)
        # value_losses.register_hook(lambda grad: scaled_value_losses)

        # Sum losses across all steps
        value_loss = value_losses.sum(dim=-1)
        reward_loss = reward_losses.sum(dim=-1)
        policy_loss = policy_losses.sum(dim=-1)

        if priorities is not None:
            # merge the first two dimensions, later split them up
            B, T, D = values.shape
            merged_values = values.reshape(B * T, D)

            pred_value_scalars = models.support_to_scalar(merged_values, self.config.support_size).detach().cpu().numpy()
            pred_value_scalars = pred_value_scalars.reshape(B, T)
            priorities = (
                    numpy.abs(pred_value_scalars - target_value_scalar)
                    ** self.config.PER_alpha
            )

        return value_loss, reward_loss, policy_loss, priorities


    def update_weights(self, batch):
        """
        Perform one training step.
        """
        double_net = self.config.network == "double"
        trans_net = self.config.network != "fully_connected" and self.config.network != "resnet"
        full_transformer = self.config.network == "transformer"

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
            mask_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = numpy.array(target_value, dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
        observation_batch = torch.tensor(numpy.array(observation_batch)).float().to(device)

        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        mask_batch = torch.tensor(mask_batch).to(device) # bool

        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(target_reward, self.config.support_size)
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1
        targets = (target_value, target_reward, target_policy)

        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )
        root_hidden_state = hidden_state

        if double_net:
            value, trans_value = torch.chunk(value, 2, dim=0)
            reward, trans_reward = torch.chunk(reward, 2, dim=0)
            policy_logits, trans_policy_logits = torch.chunk(policy_logits, 2, dim=0)
            hidden_state, root_hidden_state = torch.chunk(hidden_state, 2, dim=0)
            trans_predictions = [(trans_value, trans_reward, trans_policy_logits)]

        predictions = [(value, reward, policy_logits)]

        if full_transformer:
            # start the action batch from 1 since the first action is not used
            trans_value, trans_reward, trans_policy_logits= self.model.recurrent_inference_fast(
                root_hidden_state, action_batch[:, 1:].squeeze(-1), mask_batch
            )
            # todo, only really reward necessary
            trans_value[:, 0] = value
            trans_reward[:, 0] = reward
            trans_policy_logits[:, 0] = policy_logits
            predictions = (trans_value, trans_reward, trans_policy_logits)


        loop_length = action_batch.shape[1] if not full_transformer else 0
        for i in range(1, loop_length):
            # Instead of an action, we send the whole action sequence from start to the current action
            if trans_net:
                action_sequence = action_batch[:, 1:i].squeeze(-1)
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                    hidden_state, action_batch[:, i], action_sequence=action_sequence, root_hidden_state=root_hidden_state
                )
            else:
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                    hidden_state, action_batch[:, i]
                )

            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            if hidden_state is not None:
                hidden_state.register_hook(lambda grad: grad * 0.5)

            if double_net:
                # todo trans_pred.app(values[second_half, etc etc,
                value, trans_value = torch.chunk(value, 2, dim=0)
                reward, trans_reward = torch.chunk(reward, 2, dim=0)
                policy_logits, trans_policy_logits = torch.chunk(policy_logits, 2, dim=0)
                trans_predictions.append((trans_value, trans_reward, trans_policy_logits))

            predictions.append((value, reward, policy_logits))

        # value_loss, reward_loss, policy_loss, priorities = self.loss_loop(
        #     predictions, target_value, target_reward, target_policy,
        #     target_value_scalar, priorities, gradient_scale_batch)
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)
        if full_transformer:
            value_loss, reward_loss, policy_loss, priorities = self.loss_loop_trans(
                predictions, targets, target_value_scalar, priorities, gradient_scale_batch, mask_batch
            )
        else:
            value_loss, reward_loss, policy_loss, priorities = self.loss_loop_fast(
                predictions, targets, target_value_scalar, priorities, gradient_scale_batch)

        if double_net:
            trans_value_loss, trans_reward_loss, trans_policy_loss, _ = self.loss_loop_fast(
                trans_predictions, targets, target_value_scalar, None, gradient_scale_batch)


        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss
        if double_net:
            loss += trans_value_loss * self.config.value_loss_weight + trans_reward_loss + trans_policy_loss

        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        loss, value_loss, reward_loss, policy_loss = (
            loss.item(), value_loss.mean().item(), reward_loss.mean().item(), policy_loss.mean().item()
        )
        if double_net:
            trans_value_loss, trans_reward_loss, trans_policy_loss = (
                trans_value_loss.mean().item(), trans_reward_loss.mean().item(), trans_policy_loss.mean().item()
            )
            value_loss = (value_loss, trans_value_loss)
            reward_loss = (reward_loss, trans_reward_loss)
            policy_loss = (policy_loss, trans_policy_loss)

        return priorities, loss, value_loss, reward_loss, policy_loss

    def warmup_lr_scheduler(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)


    def update_lr(self):
        """
        Update learning rate
        """
        warmup_steps = self.config.warmup_steps
        if self.training_step < warmup_steps:
            # Linear warm-up phase
            warmup_factor = self.training_step / warmup_steps
            lr = self.config.lr_init * warmup_factor
        else:
            # Exponential decay phase
            decay_steps = max(1, self.training_step - warmup_steps)
            lr = self.config.lr_init * self.config.lr_decay_rate ** (decay_steps / self.config.lr_decay_steps)

        # Apply the learning rate to the optimizer
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=-1)(value)).sum(-1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=-1)(reward)).sum(-1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=-1)(policy_logits)).sum(-1)
        return value_loss, reward_loss, policy_loss
