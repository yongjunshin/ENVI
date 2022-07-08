import random

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class GAILActorCriticTrainer:
    def __init__(self, device, sut, state_features, action_features, history_length, lr):
        self.name = "GAIL"
        self.state_features = state_features
        self.action_features = action_features
        self.history_length = history_length

        self.lr = lr

        self.device = device
        self.sut = sut
        self.discriminator = Discriminator(self.state_features, self.action_features, self.history_length).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr/10)

        self.disc_iter = 1
        self.disc_loss = nn.MSELoss()

        self.ppo_iter = 10

        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2

    def train(self, model, epochs, tr_x, tr_y, tr_full_configs, tr_init_x, tr_init_configs, episode_length, full_logs):
        self.optimiser_pi = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in tqdm(range(epochs)):
            print(epoch, "epoch")

            # Discriminator training
            model.eval()
            self.discriminator.train()

            sim_x = tr_init_x
            model_x = []
            model_y = []
            rewards = []
            self.sut.reset()
            for sim_idx in range(episode_length):
                state_dist = model.get_distribution(sim_x)
                state = state_dist.sample().detach()
                model_x.append(sim_x)
                model_y.append(state)

                reward = self.get_reward(sim_x, state).detach()
                rewards.append(reward)

                action = self.sut.act_parallel(sim_x, state, tr_init_configs)
                next_step = torch.cat((state, action), dim=1)
                next_step = torch.reshape(next_step, (next_step.shape[0], 1, next_step.shape[1]))
                sim_x = torch.cat((sim_x, next_step), dim=1)
                sim_x = sim_x[:, 1:]

            if epoch % 50 == 0:
                sim_log = torch.stack(model_y).transpose(0, 1)
                plt.figure(figsize=(10, 5))
                plt.plot(full_logs[0, :, [0]].cpu().detach().numpy(), label="fot color")
                vis_log = np.concatenate((full_logs[0, :self.history_length, [0]].cpu().detach().numpy(),
                                          sim_log[0, :, [0]].cpu().detach().numpy()))
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

            model_x = torch.cat(model_x, dim=0)
            model_y = torch.cat(model_y, dim=0)

            print("expert judge:", self.discriminator(tr_x, tr_y).mean(), "/ model judge:", self.discriminator(model_x, model_y).mean())
            self.train_discriminator(tr_x, tr_y, model_x, model_y)

            # Model training
            model.train()
            self.discriminator.eval()

            # # BC
            # bc_dist = model.get_distribution(tr_x)
            # bc_loss = -bc_dist.log_prob(tr_y)
            # self.optimiser_pi.zero_grad()
            # bc_loss.mean().backward()
            # self.optimiser_pi.step()

            for model_epoch in range(self.ppo_iter):
                sim_x = tr_init_x
                log_probs = []
                rewards = []
                self.sut.reset()
                losses = []
                for sim_idx in range(episode_length):
                    state_dist = model.get_distribution(sim_x)
                    state = state_dist.sample().detach()
                    log_prob = state_dist.log_prob(state)
                    log_probs.append(log_prob)
                    v = model.v(sim_x)

                    reward = self.get_reward(sim_x, state).detach()
                    rewards.append(reward)

                    action = self.sut.act_parallel(sim_x, state, tr_init_configs)
                    next_step = torch.cat((state, action), dim=1)
                    next_step = torch.reshape(next_step, (next_step.shape[0], 1, next_step.shape[1]))
                    sim_x = torch.cat((sim_x, next_step), dim=1)
                    sim_x = sim_x[:, 1:]

                    delta = reward + self.gamma * model.v(sim_x) - v
                    loss = -log_prob * delta.detach() + delta*delta
                    losses.append(loss)

                self.train_policy_value_net(losses, rewards)





    def train_discriminator(self, exp_history, exp_state, model_history, model_state):
        exp_label = torch.zeros(exp_history.shape[0], device=self.device)
        model_label = torch.ones(model_history.shape[0], device=self.device)

        histories = torch.cat((exp_history, model_history), dim=0)
        states = torch.cat((exp_state, model_state), dim=0)
        labels = torch.cat((exp_label, model_label), dim=0)
        labels = torch.reshape(labels, (labels.shape[0], 1))

        disc_tr_dl = DataLoader(dataset=TensorDataset(histories, states, labels), batch_size=512, shuffle=True)

        for epoch in range(self.disc_iter):
            losses = []
            for _, (history_batch, state_batch, label_batch) in enumerate(disc_tr_dl):
                judge = self.discriminator(history_batch, state_batch)
                loss = self.disc_loss(judge, label_batch)
                losses.append(loss.item())

                self.optimiser_d.zero_grad()
                loss.backward()
                self.optimiser_d.step()
            mean_loss = np.mean(losses)
            print("Disc loss:", mean_loss)


    def train_policy_value_net(self, losses, rewards):
        loss_len = len(losses)
        losses = torch.cat(losses, dim=1)
        losses = torch.sum(losses, dim=1)
        losses = losses / loss_len
        print("Model loss:", losses.mean().item(), "Model reward:", torch.stack(rewards).mean().item())
        self.optimiser_pi.zero_grad()
        losses.mean().backward()
        self.optimiser_pi.step()


    def get_reward(self, state, action):
        reward = self.discriminator.forward(state, action)
        reward = -reward.log()
        return reward.detach()




class Discriminator(nn.Module):
    def __init__(self, state_features, action_features, history_length):
        super(Discriminator, self).__init__()
        conv_out_chan = 32
        conv_k = 3
        pool_k = 2
        self.conv1d = nn.Conv1d(in_channels=(state_features + action_features), out_channels=conv_out_chan,
                                kernel_size=conv_k)
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
