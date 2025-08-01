import copy
import time

from torch.optim.lr_scheduler import LambdaLR

import numpy
import ray
import torch

from trans_zero.utils import models
import trans_zero.networks.muzero_network as mz_net
from trans_zero.utils.other_utils import set_global_seeds
from .self_play import SelfPlay


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config
        self.previous_save_modulo = 0

        # Fix random generator seed
        set_global_seeds(self.config.seed)

        # Initialize the network
        self.model = mz_net.MuZeroNetwork(self.config)
        model_weights = copy.deepcopy(initial_checkpoint["weights"])
        try:
            self.model.set_weights(model_weights)
        except Exception as e:
            self.model.set_weights(SelfPlay.remove_module_prefix(model_weights))


        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]
        self.num_played_step = initial_checkpoint["num_played_steps"]
        self.stop_crit_step = self.training_step\
            if self.config.stopping_criterion == "training_step"\
            else self.num_played_step

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

    def save_to_shared_storage(self, shared_storage, losses, replay_buffer):
        (
            _,
            total_loss,
            value_loss,
            reward_loss,
            policy_loss,
            enc_state_loss
        ) = losses

        # Save to the shared storage
        save_modulo = self.stop_crit_step % self.config.checkpoint_interval
        save_checkpoint = save_modulo < self.previous_save_modulo
        self.previous_save_modulo = save_modulo

        if save_checkpoint:
            shared_storage.set_info.remote(
                {
                    "weights": copy.deepcopy(self.model.get_weights()),
                    "optimizer_state": copy.deepcopy(
                        models.dict_to_cpu(self.optimizer.state_dict())
                    ),
                }
            )

        if self.config.network == "double":
            value_loss, trans_value_loss = value_loss
            reward_loss, trans_reward_loss = reward_loss
            policy_loss, trans_policy_loss = policy_loss

            shared_storage.set_info.remote(
                {
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
                "enc_state_loss": enc_state_loss if enc_state_loss is not None else 0.0
            }
        )


    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop

        while self.stop_crit_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            losses = self.update_weights(batch, shared_storage)
            priorities = losses[0]

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            self.save_to_shared_storage(shared_storage, losses, replay_buffer)

            # Managing the self-play
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

    # todo should probably not be in this file and should probably happen as it is made
    def format_batch(self, batch, device):
        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
            mask_batch,
            forward_obs_batch
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)

        observation_batch = torch.tensor(numpy.array(observation_batch)).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        mask_batch = torch.tensor(mask_batch).to(device)  # bool
        if forward_obs_batch is not None:
            forward_obs_batch = torch.stack(forward_obs_batch).float().to(device)

        return observation_batch, action_batch, target_value, target_reward, target_policy, weight_batch, gradient_scale_batch, mask_batch, forward_obs_batch


    # todo consider if this should be used, then explain and test
    def calc_rep_enc_states(self, forward_obs_batch):
        B, Seq, C, H, W = forward_obs_batch.shape
        forward_obs_batch = forward_obs_batch.reshape(B * Seq, C, H, W)
        rep_enc_states = self.model.representation(forward_obs_batch)
        rep_enc_states = self.model.create_input_sequence(rep_enc_states, None)
        rep_enc_states = self.model.transformer_encoder(rep_enc_states)
        rep_enc_states = rep_enc_states.reshape(B, Seq, -1)
        return rep_enc_states


    def calc_rep_enc_state_loss(self, trans_output, rep_enc_states, weighting_factors, non_padding_mask, scaling_factors):
        enc_state_losses = self.loss_function_states(trans_output, rep_enc_states.detach())
        if self.config.loss_weight_decay is not None:
            enc_state_losses *= weighting_factors
        enc_state_losses = enc_state_losses * non_padding_mask
        enc_state_losses = enc_state_losses[:, 1:] / scaling_factors[:, 1:]
        enc_state_loss = enc_state_losses.sum(dim=1)
        return enc_state_loss



    def get_predictions_fast_trans(self, init_predictions, latent_root_state, action_batch, mask_batch):
        value, reward, policy_logits = init_predictions

        # start the action batch from 1 since the first action is not used
        trans_value, trans_reward, trans_policy_logits, transformer_output = self.model.recurrent_inference_fast(
            latent_root_state, action_batch[:, 1:].squeeze(-1), mask_batch
        )
        # set the 0th value from the inital inference
        trans_reward[:, 0] = reward
        if self.config.use_s0_for_pred:
            trans_value[:, 0] = value
            trans_policy_logits[:, 0] = policy_logits
        predictions = (trans_value, trans_reward, trans_policy_logits)

        return predictions, transformer_output


    def get_predictions(self, init_predictions, latent_root_state, action_batch, transformer_net):
        init_value, init_reward, init_policy_logits = init_predictions
        values, rewards, policy_logits_ar = [init_value], [init_reward], [init_policy_logits]
        hidden_state = latent_root_state

        for i in range(1, action_batch.shape[1]):
            # Instead of an action, we send the whole action sequence from start to the current action
            if transformer_net:
                action_sequence = action_batch[:, 1:i].squeeze(-1)
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                    None, None, action_sequence=action_sequence,
                    latent_root_state=latent_root_state
                )
            else:
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                    hidden_state, action_batch[:, i]
                )

            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            if hidden_state is not None:
                hidden_state.register_hook(lambda grad: grad * 0.5)

            values.append(value)
            rewards.append(reward)
            policy_logits_ar.append(policy_logits)

        # turn into tensors before returning
        return torch.stack(values, dim=1), torch.stack(rewards, dim=1), torch.stack(policy_logits_ar, dim=1)


    def calc_weighting_factors(self, device):
        weighting_factors = torch.tensor(
            [self.config.loss_weight_decay ** i for i in range(self.config.num_unroll_steps + 1)],
            device=device
        ).unsqueeze(0)

        # 2. Normalize
        weighting_sum = weighting_factors.sum()
        weighting_factors = weighting_factors / weighting_sum

        # 3. Scale to match the total sum of (num_unroll_steps + 1)
        n_steps = self.config.num_unroll_steps + 1
        weighting_factors = weighting_factors * n_steps
        return weighting_factors


    def weigh_losses(self, losses, weighting_factors):
        value_losses, reward_losses, policy_losses = losses

        value_losses *= weighting_factors
        reward_losses *= weighting_factors
        policy_losses *= weighting_factors

        return value_losses, reward_losses, policy_losses


    def update_weights(self, batch, shared_storage):
        """
        Perform one training step.
        """

        device = next(self.model.parameters()).device

        transformer_net = self.config.network == "transformer"
        standard_net = self.config.network == "fully_connected" or self.config.network == "resnet"

        # format batch to torch tensors
        (
            observation_batch, action_batch,
            target_value, target_reward, target_policy,
            weight_batch, gradient_scale_batch,
            mask_batch, forward_obs_batch
        ) = self.format_batch(batch, device)

        #
        target_value_scalar = numpy.array(target_value.cpu(), dtype="float32")
        priorities = numpy.zeros_like(target_value_scalar)

        target_value = models.scalar_to_support(target_value, self.config.support_size)
        target_reward = models.scalar_to_support(target_reward, self.config.support_size)

        targets = (target_value, target_reward, target_policy)

        if transformer_net: # todo
            value, reward, policy_logits, hidden_state = self.model.initial_inference(
                observation_batch,
                just_state = transformer_net and not self.config.use_s0_for_pred
            )
        else:
            value, reward, policy_logits, hidden_state = self.model.initial_inference(
                observation_batch,
            )
        init_predictions = (value, reward, policy_logits)
        latent_root_state = hidden_state

        transformer_output, rep_enc_states, enc_state_loss = None, None, None
        if self.config.encoding_loss_weight:
            rep_enc_states = self.calc_rep_enc_states(forward_obs_batch)

        # only used when doing rep enc state loss
        if transformer_net and self.config.get_fast_predictions:
            predictions, transformer_output = self.get_predictions_fast_trans(init_predictions, latent_root_state, action_batch, mask_batch)
        else:
            predictions = self.get_predictions(init_predictions, hidden_state, action_batch, transformer_net)


        if transformer_net:
            value_loss, reward_loss, policy_loss, enc_state_loss = self.loss_loop_trans(
                predictions, targets, gradient_scale_batch, mask_batch, transformer_output, rep_enc_states)
        else:
            value_loss, reward_loss, policy_loss = self.loss_loop_fast(predictions, targets, gradient_scale_batch)


        if priorities is not None:
            # merge the first two dimensions, later split them up
            values = predictions[0]
            B, T, D = values.shape
            merged_values = values.reshape(B * T, D)

            pred_value_scalars = models.support_to_scalar(merged_values, self.config.support_size).detach().cpu().numpy()
            pred_value_scalars = pred_value_scalars.reshape(B, T)
            priorities = (
                    numpy.abs(pred_value_scalars - target_value_scalar)
                    ** self.config.PER_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        reward_loss_weight = 1 if self.config.predict_reward else 0
        loss = value_loss * self.config.value_loss_weight + reward_loss * reward_loss_weight + policy_loss

        if enc_state_loss is not None:
            loss += (enc_state_loss * self.config.encoding_loss_weight)
            # for plotting
            enc_state_loss = enc_state_loss.mean().item()

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
        self.num_played_step = ray.get(
            shared_storage.get_info.remote("num_played_steps")
        )

        self.stop_crit_step = self.training_step\
            if self.config.stopping_criterion == "training_step"\
            else self.num_played_step

        loss, value_loss, reward_loss, policy_loss = (
            loss.item(), value_loss.mean().item(), reward_loss.mean().item(), policy_loss.mean().item()
        )

        return priorities, loss, value_loss, reward_loss, policy_loss, enc_state_loss

# todo, consider loss class
    def loss_loop_trans(self, predictions, targets, gradient_scale_batch, action_mask, trans_output = None, rep_enc_states = None):
        (values, rewards, policy_logits) = predictions
        (target_value, target_reward, target_policy) = targets

        # Compute losses
        value_losses, reward_losses, policy_losses = self.loss_function(
            values, rewards, policy_logits, target_value, target_reward, target_policy
        )

        # Generate exponentially decreasing weights
        weighting_factors = None
        if self.config.loss_weight_decay is not None:
            weighting_factors = self.calc_weighting_factors(values.device)
            value_losses, reward_losses, policy_losses = self.weigh_losses(
                (value_losses, reward_losses, policy_losses), weighting_factors
            )

        reward_losses[:,0] = 0.0

        non_padding_mask = ~action_mask
        # Apply mask to losses
        # todo consider if you should really remove losses after game ended
        # value_losses = value_losses * non_padding_mask
        # reward_losses = reward_losses * non_padding_mask
        # policy_losses = policy_losses * non_padding_mask

        # Directly scale the losses, add an epsilon to avoid dividing by zero
        scaling_factors = gradient_scale_batch + 1e-8  # Shape: (batch_size, sequence_length)

        # Apply scaling from index 1 onwards
        value_losses[:, 1:] = value_losses[:, 1:] / scaling_factors[:, 1:]
        reward_losses[:, 1:] = reward_losses[:, 1:] / scaling_factors[:, 1:]
        policy_losses[:, 1:] = policy_losses[:, 1:] / scaling_factors[:, 1:]

        value_loss = value_losses.sum(dim=1)  # Shape: (batch_size,)
        reward_loss = reward_losses.sum(dim=1)
        policy_loss = policy_losses.sum(dim=1)

        enc_state_loss = None
        if rep_enc_states is not None:
            enc_state_loss = self.calc_rep_enc_state_loss(trans_output, rep_enc_states, weighting_factors, non_padding_mask, scaling_factors)

        return value_loss, reward_loss, policy_loss, enc_state_loss


    def loss_loop_fast(self, predictions, targets, gradient_scale_batch):

        (target_value, target_reward, target_policy) = targets
        (values, rewards, policy_logits_ar) = predictions

        # Compute losses
        value_losses, reward_losses, policy_losses = self.loss_function(
            values, rewards, policy_logits_ar, target_value, target_reward, target_policy
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

        return value_loss, reward_loss, policy_loss


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
        if self.stop_crit_step < warmup_steps:
            # Linear warm-up phase
            warmup_factor = self.stop_crit_step / warmup_steps
            lr = self.config.lr_init * warmup_factor
        else:
            # Exponential decay phase
            decay_steps = max(1, self.stop_crit_step - warmup_steps)
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
        value_loss = (-target_value * torch.nn.LogSoftmax(dim=-1)(value)).sum(-1)
        reward_loss = (-target_reward * torch.nn.LogSoftmax(dim=-1)(reward)).sum(-1)
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=-1)(policy_logits)).sum(-1)
        return value_loss, reward_loss, policy_loss

    @staticmethod
    def loss_function_states(
        enc_state,
        target_enc_state,
    ):
        # MSE
        # Compute the MSE loss by summing squared differences over the last dimension
        state_loss = ((enc_state - target_enc_state) ** 2).sum(-1)
        return state_loss

    # todo consider if relevant to fix later
    # def update_weights_with_double_net(self, batch):
    #     """
    #     Perform one training step.
    #     """
    #
    #     double_net = self.config.network == "double"
    #     trans_net = self.config.network != "fully_connected" and self.config.network != "resnet"
    #     full_transformer = self.config.network == "transformer"
    #
    #     (
    #         observation_batch,
    #         action_batch,
    #         target_value,
    #         target_reward,
    #         target_policy,
    #         weight_batch,
    #         gradient_scale_batch,
    #         mask_batch,
    #         forward_obs_batch
    #     ) = batch
    #
    #     # Keep values as scalars for calculating the priorities for the prioritized replay
    #     target_value_scalar = numpy.array(target_value, dtype="float32")
    #     priorities = numpy.zeros_like(target_value_scalar)
    #
    #     device = next(self.model.parameters()).device
    #     if self.config.PER:
    #         weight_batch = torch.tensor(weight_batch.copy()).float().to(device)
    #     observation_batch = torch.tensor(numpy.array(observation_batch)).float().to(device)
    #
    #     action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
    #     target_value = torch.tensor(target_value).float().to(device)
    #     target_reward = torch.tensor(target_reward).float().to(device)
    #     target_policy = torch.tensor(target_policy).float().to(device)
    #     gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
    #     mask_batch = torch.tensor(mask_batch).to(device) # bool
    #     if forward_obs_batch is not None:
    #         forward_obs_batch = torch.stack(forward_obs_batch).float().to(device)
    #
    #     # observation_batch: batch, channels, height, width
    #     # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
    #     # target_value: batch, num_unroll_steps+1
    #     # target_reward: batch, num_unroll_steps+1
    #     # target_policy: batch, num_unroll_steps+1, len(action_space)
    #     # gradient_scale_batch: batch, num_unroll_steps+1
    #
    #     target_value = models.scalar_to_support(target_value, self.config.support_size)
    #     target_reward = models.scalar_to_support(target_reward, self.config.support_size)
    #     # target_value: batch, num_unroll_steps+1, 2*support_size+1
    #     # target_reward: batch, num_unroll_steps+1, 2*support_size+1
    #     targets = (target_value, target_reward, target_policy)
    #
    #     value, reward, policy_logits, hidden_state = self.model.initial_inference(
    #         observation_batch
    #     )
    #     latent_root_state = hidden_state
    #
    #     if double_net:
    #         value, trans_value = torch.chunk(value, 2, dim=0)
    #         reward, trans_reward = torch.chunk(reward, 2, dim=0)
    #         policy_logits, trans_policy_logits = torch.chunk(policy_logits, 2, dim=0)
    #         hidden_state, latent_root_state = torch.chunk(hidden_state, 2, dim=0)
    #         trans_predictions = [(trans_value, trans_reward, trans_policy_logits)]
    #
    #     predictions = [(value, reward, policy_logits)]
    #
    #     if full_transformer:
    #         # start the action batch from 1 since the first action is not used
    #         trans_value, trans_reward, trans_policy_logits, transformer_output = self.model.recurrent_inference_fast(
    #             latent_root_state, action_batch[:, 1:].squeeze(-1), mask_batch
    #         )
    #         # todo, only really reward necessary
    #         trans_value[:, 0] = value
    #         trans_reward[:, 0] = reward
    #         trans_policy_logits[:, 0] = policy_logits
    #         predictions = (trans_value, trans_reward, trans_policy_logits)
    #
    #         if self.config.encoding_loss_weight:
    #             B, Seq, C, H, W = forward_obs_batch.shape
    #             forward_obs_batch = forward_obs_batch.reshape(B*Seq, C, H, W)
    #             rep_enc_states = self.model.representation(forward_obs_batch)
    #             rep_enc_states = self.model.create_input_sequence(rep_enc_states, None)
    #             rep_enc_states = self.model.transformer_encoder(rep_enc_states)
    #
    #             rep_enc_states = rep_enc_states.reshape(B, Seq, -1)
    #         else:
    #             rep_enc_states = None
    #
    #
    #
    #     loop_length = action_batch.shape[1] if not full_transformer else 0
    #     for i in range(1, loop_length):
    #         # Instead of an action, we send the whole action sequence from start to the current action
    #         if trans_net:
    #             action_sequence = action_batch[:, 1:i].squeeze(-1)
    #             value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
    #                 hidden_state, action_batch[:, i], action_sequence=action_sequence, latent_root_state=latent_root_state
    #             )
    #         else:
    #             value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
    #                 hidden_state, action_batch[:, i]
    #             )
    #
    #         # Scale the gradient at the start of the dynamics function (See paper appendix Training)
    #         if hidden_state is not None:
    #             hidden_state.register_hook(lambda grad: grad * 0.5)
    #
    #         if double_net:
    #             # todo trans_pred.app(values[second_half, etc etc,
    #             value, trans_value = torch.chunk(value, 2, dim=0)
    #             reward, trans_reward = torch.chunk(reward, 2, dim=0)
    #             policy_logits, trans_policy_logits = torch.chunk(policy_logits, 2, dim=0)
    #             trans_predictions.append((trans_value, trans_reward, trans_policy_logits))
    #
    #         predictions.append((value, reward, policy_logits))
    #
    #     # value_loss, reward_loss, policy_loss, priorities = self.loss_loop(
    #     #     predictions, target_value, target_reward, target_policy,
    #     #     target_value_scalar, priorities, gradient_scale_batch)
    #     # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)
    #     enc_state_loss = None
    #     if full_transformer:
    #         value_loss, reward_loss, policy_loss, priorities, enc_state_loss = self.loss_loop_trans(
    #             predictions, targets, target_value_scalar, priorities, gradient_scale_batch, mask_batch,
    #             transformer_output, rep_enc_states)
    #
    #     else:
    #         value_loss, reward_loss, policy_loss, priorities = self.loss_loop_fast(
    #             predictions, targets, target_value_scalar, priorities, gradient_scale_batch)
    #
    #     if double_net:
    #         trans_value_loss, trans_reward_loss, trans_policy_loss, _ = self.loss_loop_fast(
    #             trans_predictions, targets, target_value_scalar, None, gradient_scale_batch)
    #
    #
    #     # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
    #     reward_loss_weight = 1 if self.config.predict_reward else 0
    #     loss = value_loss * self.config.value_loss_weight + reward_loss * reward_loss_weight + policy_loss
    #
    #     if double_net:
    #         loss += trans_value_loss * self.config.value_loss_weight + trans_reward_loss * reward_loss_weight + trans_policy_loss
    #
    #     if enc_state_loss is not None:
    #         loss += (enc_state_loss * self.config.encoding_loss_weight)
    #         # for plotting
    #         enc_state_loss = enc_state_loss.mean().item()
    #
    #     if self.config.PER:
    #         # Correct PER bias by using importance-sampling (IS) weights
    #         loss *= weight_batch
    #     # Mean over batch dimension (pseudocode do a sum)
    #     loss = loss.mean()
    #
    #     # Optimize
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     self.training_step += 1
    #
    #     loss, value_loss, reward_loss, policy_loss = (
    #         loss.item(), value_loss.mean().item(), reward_loss.mean().item(), policy_loss.mean().item()
    #     )
    #     if double_net:
    #         trans_value_loss, trans_reward_loss, trans_policy_loss = (
    #             trans_value_loss.mean().item(), trans_reward_loss.mean().item(), trans_policy_loss.mean().item()
    #         )
    #         value_loss = (value_loss, trans_value_loss)
    #         reward_loss = (reward_loss, trans_reward_loss)
    #         policy_loss = (policy_loss, trans_policy_loss)
    #
    #     return priorities, loss, value_loss, reward_loss, policy_loss, enc_state_loss