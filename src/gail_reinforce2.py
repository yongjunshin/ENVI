import random

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class GAILReinforceTrainer:
    def __init__(self, device, sut, state_features, action_features, history_length, lr):
        self.name = "GAIL"
        self.state_features = state_features
        self.action_features = action_features
        self.history_length = history_length

        self.lr = lr

        self.device = device
        self.sut = sut
        self.discriminator = Discriminator(self.state_features, self.action_features, self.history_length).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr*10)

        self.disc_iter = 10
        self.disc_loss = nn.MSELoss()

        self.ppo_iter = 10

        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2

    def train(self, model, epochs, tr_x, tr_y, tr_full_configs, tr_init_x, tr_init_configs, episode_length, full_logs) -> list:
        self.optimiser_pi = torch.optim.Adam(model.parameters(), lr=self.lr)
        gail_episode_length = episode_length

        evaluation_results = []


        tr_dl = DataLoader(dataset=TensorDataset(tr_x, tr_y, tr_full_configs), batch_size=10, shuffle=True)
        #testing_dl = DataLoader(dataset=TensorDataset(xt, yt), batch_size=512, shuffle=True)

        # initial model
        #evaluation_results.append(simulation_and_comparison_with_multiple_testing_dataset(model, self.sut, xt, yt, self.device, self.name+"0"))

        reward_list = []
        for epch in tqdm(range(epochs), desc="Training"):
            print("epoch:", epch)
            ed_sum = torch.zeros((), device=self.device)
            dtw_sum = torch.zeros((), device=self.device)

            if epch % 10 == 0:
                # Discriminator training
                model.eval()
                self.discriminator.train()

                pi_states = []
                pi_actions = []
                sim_x = tr_init_x

                line_tracer_idx, _ = torch.max(torch.abs(sim_x[:, :, [1]]), dim=1)
                line_tracer_idx = line_tracer_idx.cpu().numpy()

                self.sut.reset()
                for sim_idx in range(episode_length):
                    pi_states.append(sim_x)
                    # action choice
                    action_prob = model.get_distribution(sim_x)
                    action = action_prob.sample().detach()
                    pi_actions.append(action)
                    if torch.any(torch.lt(action, -3)):
                        print()

                    # state transition
                    sys_operations = self.sut.act_parallel(sim_x, action, tr_init_configs)
                    # sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                    next_x = torch.cat((action, sys_operations), dim=1)
                    next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                    sim_x = sim_x[:, 1:]
                    sim_x = torch.cat((sim_x, next_x), dim=1)


                if epch % 1 == 0:
                    sim_log = torch.stack(pi_actions).transpose(0, 1)
                    plt.figure(figsize=(10, 5))
                    plt.plot(full_logs[0, :, [0]].cpu().detach().numpy(), label="fot color")
                    vis_log = np.concatenate((full_logs[0, :self.history_length, [0]].cpu().detach().numpy(), sim_log[0, :, [0]].cpu().detach().numpy()))
                    plt.plot(vis_log, label="simul color")
                    plt.legend()
                    plt.show()
                    plt.figure(figsize=(10, 5))
                    plt.plot(full_logs[0, :, [1]].cpu().detach().numpy(), label="fot dist")
                    vis_log = np.concatenate((full_logs[0, :self.history_length, [1]].cpu().detach().numpy(),
                                              sim_log[0, :, [1]].cpu().detach().numpy()))
                    plt.plot(vis_log, label="simul dist")
                    plt.legend()
                    plt.show()


                pi_states = torch.cat(pi_states)
                pi_actions = torch.cat(pi_actions)




                print("before training")
                print("expert judge: ", self.discriminator(tr_x, tr_y).mean(), "model judge: ",
                      self.discriminator(pi_states, pi_actions).mean())
                print("expert reward: ", self.get_reward(tr_x, tr_y).mean(), "model reward: ",
                      self.get_reward(pi_states, pi_actions).mean())

                self.train_discriminator(tr_x, tr_y, pi_states, pi_actions)

                print("after training")
                print("expert judge: ", self.discriminator(tr_x, tr_y).mean(), "model judge: ",
                      self.discriminator(pi_states, pi_actions).mean())
                print("expert reward: ", self.get_reward(tr_x, tr_y).mean(), "model reward: ",
                      self.get_reward(pi_states, pi_actions).mean())




            # Policy training
            model.train()
            self.discriminator.eval()
            self.sut.reset()

            sim_x = tr_init_x
            y_pred = torch.zeros((sim_x.shape[0], gail_episode_length, sim_x.shape[2]), device=self.device)

            rewards = []
            probs = []
            states = []
            states_prime = []
            actions = []
            reward_mean_accum = 0
            for sim_idx in range(gail_episode_length):
                # action choice
                action_distribution = model.get_distribution(sim_x)
                states.append(sim_x)
                action = action_distribution.sample().detach()
                actions.append(action)
                prob = action_distribution.log_prob(action)
                probs.append(prob)

                # get reward
                reward = self.get_reward(sim_x, action).detach()
                reward_mean_accum = reward_mean_accum + reward.mean().item()
                rewards.append(reward)

                # state transition
                sys_operations = self.sut.act_parallel(sim_x, action, tr_init_configs)
                # sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
                next_x = torch.cat((action, sys_operations), dim=1)
                next_x = torch.reshape(next_x, (next_x.shape[0], 1, next_x.shape[1]))
                sim_x = sim_x[:, 1:]
                sim_x = torch.cat((sim_x, next_x), dim=1)
                y_pred[:, sim_idx] = sim_x[:, -1]
                states_prime.append(sim_x)

            print("reward mean:", reward_mean_accum)
            self.train_policy_value_net(model, states, states_prime, actions, probs, rewards)
            reward_list.append(reward_mean_accum)


            # if epch%20 == 19:
            #     figureName = self.name + str(epch + 1)
            # else:
            #     figureName = None
            # #evaluation_results.append(simulation_and_comparison_with_multiple_testing_dataset(model, self.sut, xt, yt, self.device, figureName))
        return reward_list

    def train_discriminator(self, exp_state, exp_action, pi_states, pi_actions):
        index_list = list(range(len(pi_states)))
        random.shuffle(index_list)
        index_list = index_list[:int(len(exp_state)/2)]
        exp_state = exp_state[index_list]
        exp_action = exp_action[index_list]
        noise_color = torch.randn(exp_state[:,:,[0]].shape, device=self.device) * 0.01 - 0.005
        noise_dist = torch.randn(exp_state[:,:,[1]].shape, device=self.device) * 0.001 - 0.0005
        noise = torch.cat([noise_color, noise_dist], dim=2)
        noise = torch.cat([noise, noise], dim=2)
        # plt.figure(figsize=(10, 5))
        # plt.plot(exp_state[0, :, [3]].cpu().detach().numpy(), label="fot color")
        # plt.plot((exp_state + noise)[0, :, [3]].cpu().detach().numpy(), label="fot color + noise")
        # plt.legend()
        # plt.show()
        exp_state = exp_state + noise


        pi_states = pi_states[index_list]
        pi_actions = pi_actions[index_list]

        states = torch.cat([exp_state, pi_states], dim=0)
        actions = torch.cat([exp_action, pi_actions], dim=0)
        exp_trajectory_label = torch.zeros(len(exp_action))
        pi_trajectory_label = torch.ones(len(pi_actions))
        labels = torch.cat((exp_trajectory_label, pi_trajectory_label)).to(device=self.device).type(torch.float32)
        labels = torch.reshape(labels, (labels.shape[0], 1))

        tr_dl = DataLoader(dataset=TensorDataset(states, actions, labels), batch_size=512, shuffle=True)
        index_list = list(range(len(states)))

        for i in range(self.disc_iter):
            loss_accum = 0
            for _, (state_batch, action_batch, label_batch) in enumerate(tr_dl):
                judges = self.discriminator(state_batch, action_batch)
                loss = self.disc_loss(judges, label_batch)
                loss_accum = loss_accum + loss.item()

                self.optimiser_d.zero_grad()
                loss.backward()
                self.optimiser_d.step()
            print("disc loss epoch,", i, ":", loss_accum)

    def train_policy_value_net(self, model, states, states_prime, actions, probs, rewards):
        R = torch.zeros(rewards[0].shape, device=self.device)

        i = 0
        prob_len = len(probs)
        self.optimiser_pi.zero_grad()
        for reward in rewards[::-1]:
            R = reward + self.gamma * R
            loss = -probs[prob_len - i - 1] * R
            loss.mean().backward()
            i = i + 1
            # make_dot(loss.mean(), params=dict(model.named_parameters())).render(
            #      "graph", format="png")
        # print("loss", loss.mean())
        self.optimiser_pi.step()


    def get_reward(self, state, action):
        reward = self.discriminator.forward(state, action)
        reward = -reward.log()
        reward = torch.nan_to_num(reward)
        #reward = torch.tanh(reward)
        return reward.detach()


class Discriminator(nn.Module):
    def __init__(self, state_features, action_features, history_length):
        super(Discriminator, self).__init__()
        conv_out_chan = 32
        conv_k = 3
        pool_k = 2
        self.conv1d = nn.Conv1d(in_channels=(state_features + action_features), out_channels=conv_out_chan, kernel_size=conv_k)
        conv_l_out = history_length - conv_k + 1
        self.max_pool1d = nn.MaxPool1d(kernel_size=pool_k)
        pool_l_out = int((conv_l_out - pool_k) / pool_k + 1)
        hidden_features = 512
        self.fc_input = nn.Linear(conv_out_chan * pool_l_out + state_features, hidden_features)
        self.fc_middle_1 = nn.Linear(hidden_features, hidden_features)
        self.fc_middle_2 = nn.Linear(hidden_features, hidden_features)
        self.fc_middle_3 = nn.Linear(hidden_features, hidden_features)
        self.fc_output = nn.Linear(hidden_features, 1)

    def forward(self, history, state):
        x = torch.swapaxes(history, 1, 2)
        out_history = torch.torch.relu(torch.nan_to_num(self.conv1d(x)))
        out_history = self.max_pool1d(out_history)
        out_history = torch.flatten(out_history, start_dim=1)
        out = torch.cat([out_history, state], dim=1)
        out = torch.nan_to_num(self.fc_input(out))
        out = torch.relu(out)
        out = torch.nan_to_num(self.fc_middle_1(out))
        out = torch.relu(out)
        out = torch.nan_to_num(self.fc_middle_2(out))
        out = torch.relu(out)
        out = torch.nan_to_num(self.fc_middle_3(out))
        out = torch.relu(out)
        out = torch.nan_to_num(self.fc_output(out))
        out = torch.sigmoid(out)
        return out
