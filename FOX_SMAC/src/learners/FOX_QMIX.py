import copy
import torch as th
import numpy as np
import torch.nn.functional as F
import random

from torch.optim import RMSprop
from modules.mixers.qmix import QMixer
from components.episode_buffer import EpisodeBatch
from modules.FOX.FNet import Encoder, Decoder, DecoderTau, VAEModel


class FOX_QMIX:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = QMixer(args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.encoder = Encoder(args.rnn_hidden_dim, args.predict_net_dim, 64)
        self.decoder = Decoder(64*3, args.predict_net_dim, args.n_agents * 4)
        self.decoder_tau = DecoderTau(64, args.predict_net_dim, 64)
        self.VAEModel = VAEModel(Encoder = self.encoder, Decoder = self.decoder, DecoderTau = self.decoder_tau)
        self.diversity_mean = 0
        self.normalizer = 0
        
        self.iteration = 25

        if self.args.use_cuda:

            self.encoder.to(th.device(self.args.GPU))
            self.decoder.to(th.device(self.args.GPU))
            self.decoder_tau.to(th.device(self.args.GPU))
            self.VAEModel.to(th.device(self.args.GPU))

        self.optimiser = RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.list = [(np.arange(args.n_agents - i) + i).tolist() + np.arange(i).tolist()
                     for i in range(args.n_agents)]

    def train_predict(self, batch: EpisodeBatch, t_env: int):

        # Get the relevant quantities
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions

        max_idx = batch["max_idx"][:, :-1]
        min_idx = batch["min_idx"][:, :-1]

        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)
        initial_hidden = self.mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

        _, hidden_store, _, _ = self.mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach(), self.VAEModel.Encoder)
        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        # current formations
        formation = batch["formation"][:, :-1]

        history = hidden_store[:, :-1].clone().detach()

        full_loss_list = []
        
        # random sampling for FNet Training.
        random_batch = [random.randrange(batch.batch_size) for _ in range(self.iteration)]
        random_time = [random.randrange(batch.max_seq_length - 1) for _ in range(self.iteration)]
        random_agent = [random.randrange(self.args.n_agents) for _ in range(self.iteration)]

        for epoch in range(self.iteration):
            tau = history[random_batch[epoch], random_time[epoch]]

            idx_max = int(max_idx[random_batch[epoch], random_time[epoch], random_agent[epoch]].item())
            idx_min = int(min_idx[random_batch[epoch], random_time[epoch], random_agent[epoch]].item())

            z_i, mean_i, log_var_i = self.VAEModel.Encoder(tau[random_agent][epoch])
            z_max, mean_max, log_var_max = self.VAEModel.Encoder(tau[idx_max])
            z_min, mean_min, log_var_min = self.VAEModel.Encoder(tau[idx_min])

            full_loss = self.VAEModel.update(tau[random_agent][epoch],z_i,z_max,z_min,formation[random_batch[epoch],random_time[epoch]],
                                             log_var_i, log_var_max, log_var_min,
                                             mean_i, mean_max, mean_min)

            if full_loss is not None:
                full_loss_list.append(full_loss)

            self.logger.log_stat("full_loss", np.array(
                full_loss_list).mean(), t_env)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        th.set_printoptions(threshold = 10_000)

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]
        last_actions_onehot = th.cat([th.zeros_like(
            actions_onehot[:, 0].unsqueeze(1)), actions_onehot], dim=1)  # last_actions

        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)
        initial_hidden = self.mac.hidden_states.clone().detach()
        initial_hidden = initial_hidden.reshape(
            -1, initial_hidden.shape[-1]).to(self.args.device)
        # [batch, agent, seq_length, obs]
        input_here = th.cat((batch["obs"], last_actions_onehot),
                            dim=-1).permute(0, 2, 1, 3).to(self.args.device)

        # hidden_store = [batch x agent, seq, 64]]
        mac_out, hidden_store, local_qs,q_f = self.mac.agent.forward(
            input_here.clone().detach(), initial_hidden.clone().detach(), self.VAEModel.Encoder)

        hidden_store = hidden_store.reshape(
            -1, input_here.shape[1], hidden_store.shape[-2], hidden_store.shape[-1]).permute(0, 2, 1, 3)

        formation = batch["formation"][:, 1:]
        max_idx = batch["max_idx"][:, :-1]
        min_idx = batch["min_idx"][:, :-1]
        history = hidden_store[:, 1:].clone().detach()

        visit = batch["visit"][:,1:]

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals -
                      chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        initial_hidden_target = self.target_mac.hidden_states.clone().detach()
        initial_hidden_target = initial_hidden_target.reshape(
            -1, initial_hidden_target.shape[-1]).to(self.args.device)
        target_mac_out, _, _, _ = self.target_mac.agent.forward(
            input_here.clone().detach(), initial_hidden_target.clone().detach(), self.VAEModel.Encoder)
        target_mac_out = target_mac_out[:, 1:]

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(
                target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Intrinsic

        with th.no_grad():

            visit_reward = th.where(visit != 0, 1 / th.sqrt(visit), th.zeros_like(visit))

            max_visit = th.max(visit_reward).item()
            normalized_visit = visit_reward - max_visit

            formation_repeat = formation.clone().detach().unsqueeze(2).repeat(1,1,20,1)

            diversity_reward = 0

            batch_size = batch.batch_size
            max_seq_length = batch.max_seq_length

            for agent in range(self.args.n_agents):
                tau = history[:, :, agent, :]

                idx_max = max_idx[:, :, agent].detach().long()
                idx_min = min_idx[:, :, agent].detach().long()

                tau_max = history[th.arange(batch_size)[:, None], idx_max, agent]
                tau_min = history[th.arange(batch_size)[:, None], idx_min, agent]

                tau = tau.unsqueeze(2).expand(-1, -1, 20, -1)
                tau_max = tau_max.unsqueeze(2).expand(-1, -1, 20, -1)
                tau_min = tau_min.unsqueeze(2).expand(-1, -1, 20, -1)

                z_rand1 = th.randn(batch_size, max_seq_length - 1, 20, 64, device=self.args.GPU)
                z_rand2 = th.randn(batch_size, max_seq_length - 1, 20, 64, device=self.args.GPU)

                # get z for agents in F^i
                z_i, _, _ = self.VAEModel.Encoder(tau)
                z_max, _, _ = self.VAEModel.Encoder(tau_max)
                z_min, _, _ = self.VAEModel.Encoder(tau_min)

                # q(F)
                f_prime_z = self.VAEModel(z_i,z_max,z_min)
                # p(F)
                f_prime_i = self.VAEModel(z_i,z_rand1,z_rand2)
                f_prime_max = self.VAEModel(z_rand1, z_max, z_rand2)
                f_prime_min = self.VAEModel(z_rand1, z_rand2, z_min)

                log_q_o = self.VAEModel.get_log_pi(f_prime_z, formation_repeat)

                log_p_i_o = self.VAEModel.get_log_pi(f_prime_i, formation_repeat)
                log_p_max_o = self.VAEModel.get_log_pi(f_prime_max, formation_repeat)
                log_p_min_o = self.VAEModel.get_log_pi(f_prime_min, formation_repeat)

                nan_mask = th.isnan(log_q_o)
                log_q_o[nan_mask] = 0
                mean_log_q_o= log_q_o.sum() / nan_mask.size(0) - nan_mask.sum()
                self.diversity_mean = max(self.diversity_mean, abs(mean_log_q_o))
                self.normalizer = abs(mean_log_q_o)/self.diversity_mean

                ir = log_q_o - log_p_i_o / 3 - log_p_max_o / 3 - log_p_min_o / 3

                max_ir = th.max(ir).item()
                min_ir = th.min(ir).item()

                ir = ((ir - max_ir) / (max_ir - min_ir + 1e-6)) * self.normalizer

                ir = ir.mean(dim=2)

                diversity_reward += ir * self.args.beta2

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(
                chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(
                target_max_qvals, batch["state"][:, 1:])

        targets = rewards + diversity_reward + self.args.beta1 * normalized_visit + self.args.gamma * (1 - terminated) * target_max_qvals

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            if self.mixer == None:
                tot_q_data = np.mean(tot_q_data, axis=2)
                tot_target = np.mean(tot_target, axis=2)

            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()
        update_prior = (masked_td_error ** 2).squeeze().sum(dim=-1,
                                                            keepdim=True) / mask.squeeze().sum(dim=-1, keepdim=True)

        norm_loss = F.l1_loss(local_qs, target=th.zeros_like(
            local_qs), reduction='none')[:, :-1]
        mask_expand = mask.unsqueeze(-1).expand_as(norm_loss)
        norm_loss = (norm_loss * mask_expand).sum() / mask_expand.sum()
        loss += 0.1 * norm_loss # lambda = 0.1

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
                                 mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(
                "target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self.logger.log_stat("rewards", th.mean(rewards), t_env)
            self.logger.log_stat("diversity_reward", th.mean(diversity_reward), t_env)
            self.logger.log_stat("visit_reward", th.mean(self.args.beta1 * normalized_visit), t_env)
            self.logger.log_stat("total_reward", th.mean(rewards + diversity_reward + self.args.beta1 * normalized_visit), t_env)
            self.log_stats_t = t_env

        return update_prior.squeeze().detach()

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(th.device(self.args.GPU))
            self.target_mixer.to(th.device(self.args.GPU))

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(
            th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
