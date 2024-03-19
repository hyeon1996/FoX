from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import csv
import os
from collections import defaultdict

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        self.formation = defaultdict(int)

        self.env_info = self.get_env_info()
        self.obs_size = self.env_info["obs_shape"]
        self.trans = np.random.rand(9, self.obs_size)

        self.round = self.args.round

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        if self.args.mac == "oda_mac":
            self.mac.init_latent(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            batch_max_idx = []
            batch_min_idx = []
            dlist = []

            obs = pre_transition_data["obs"][0]

            # transform obs to formation
            for i in range(self.args.n_agents):
                obs1 = pre_transition_data["obs"][0][i]
                diff = obs1 - obs
                dist = np.sqrt(np.sum(np.square(diff), axis = 1))
                dist = np.round(dist,self.round)

                dist_temp = np.copy(dist)
                dist_temp[i] = np.inf

                max_idx = np.argmax(dist)
                min_idx = np.argmin(dist_temp)

                max_diff = obs1 - obs[max_idx]
                min_diff = obs1 - obs[min_idx]

                batch_max_idx.append(max_idx)
                batch_min_idx.append(min_idx)

                max_key = np.matmul(self.trans, max_diff)
                min_key = np.matmul(self.trans, min_diff)

                max_binary = "".join(str(int(max(np.sign(d), 0))) for d in max_key)
                min_binary = "".join(str(int(max(np.sign(d), 0))) for d in min_key)

                max_key = int(max_binary, 2)
                min_key = int(min_binary, 2)

                dlist.extend([dist[max_idx], dist[min_idx], max_key, min_key])

            klist = tuple(dlist)

            
            if not test_mode:
                self.formation[klist] += 1

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    "formation": [klist],
                    "max_idx": [batch_max_idx],
                    "min_idx": [batch_min_idx],
                    "visit": [(self.formation[klist],)],
                }
            else:
                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    "formation": [klist],
                    "max_idx": [batch_max_idx],
                    "min_idx": [batch_min_idx],
                    "visit": [(0,)],
                }
            
            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)
                         for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            cur_returns_mean = np.array(
                [0 if item <= 0 else 1 for item in cur_returns]).mean()
            #self.writereward(cur_returns_mean, self.t_env)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean",
                             np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std",
                             np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v/stats["n_episodes"], self.t_env)
        stats.clear()
