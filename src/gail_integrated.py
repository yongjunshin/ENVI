import random

from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.data_manager import tensor_log_to_tensor_xys
from src.envi_util import evaluate_model, save_model
from src.simulation import non_deterministic_simulation


class GAILIntegration:
    def __init__(self, name, device, sut, state_features, action_features, history_length, lr, disc_learning_break, disc_iter, ppo_iter, deterministic=False):
        self.name = name
        self.state_features = state_features
        self.action_features = action_features
        self.history_length = history_length
        self.loss_fn = torch.nn.MSELoss()

        self.lr = lr

        self.device = device
        self.sut = sut
        self.discriminator = Discriminator(self.state_features, self.action_features, self.history_length).to(device=self.device)

        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.disc_learning_break = disc_learning_break
        self.disc_iter = disc_iter
        self.disc_loss = nn.MSELoss()

        self.ppo_iter = ppo_iter

        self.gamma = 0.99
        self.lmbda = 0.95
        self.eps_clip = 0.2

        self.deterministic_flag = deterministic

    def train(self, model, epochs, tr_dataset, model_dir_path, bc=False, algorithm="ppo", model_save_period=1):
        self.optimiser_pi = torch.optim.Adam(model.parameters(), lr=self.lr)

        tr_losses = []
        va_losses = []
        tr_euclidean_losses = []
        va_euclidean_losses = []
        tr_dtw_losses = []
        va_dtw_losses = []

        model_save_epochs = []
        for epch in tqdm(range(epochs), desc="Training"):
            if epch % (1 + self.disc_learning_break) == 0:
                self.update_discriminator(model, tr_dataset)

            if bc:
                self.bc_pre_training(model, tr_dataset)
            self.update_model(model, tr_dataset, algorithm)

            if epch % model_save_period == 0:
                model_save_epochs.append(epch)

                tr_loss, va_loss, tr_euclidean_loss, va_euclidean_loss, tr_dtw_loss, va_dtw_loss = \
                    evaluate_model(self, model, self.deterministic_flag, tr_dataset, self.name)
                tr_losses.append(tr_loss)
                va_losses.append(va_loss)
                tr_euclidean_losses.append(tr_euclidean_loss)
                va_euclidean_losses.append(va_euclidean_loss)
                tr_dtw_losses.append(tr_dtw_loss)
                va_dtw_losses.append(va_dtw_loss)

                save_model(model, model_dir_path, self.name, epch)

        return model_save_epochs, tr_losses, va_losses, tr_euclidean_losses, va_euclidean_losses, tr_dtw_losses, va_dtw_losses

    def bc_pre_training(self, model, tr_dataset):
        model.train()
        tr_x = tr_dataset.tr_tensor_x
        tr_y = tr_dataset.tr_tensor_y
        tr_dl = DataLoader(dataset=TensorDataset(tr_x, tr_y), batch_size=512, shuffle=True)

        for _, (x_batch, y_batch) in enumerate(tr_dl):
            # non-deterministic
            bc_dist = model.get_distribution(x_batch)
            loss = -bc_dist.log_prob(y_batch)
            loss = loss.mean()

            self.optimiser_pi.zero_grad()
            loss.backward()
            self.optimiser_pi.step()

    def update_model(self, model, tr_dataset, algorithm):
        model.train()
        self.discriminator.eval()

        histories, histories_prime, states, state_log_probs, rewards, ac_losses = self.collect_trajectory(model, tr_dataset)

        if algorithm == 'reinforce':
            self.train_model_by_reinforce(state_log_probs, rewards)
        elif algorithm == 'actor_critic':
            self.train_model_by_actor_critic(ac_losses)
        elif algorithm == 'ppo':
            self.train_model_by_ppo(model, histories, histories_prime, states, state_log_probs, rewards)
        else:
            print("WRONG RL ALGORITHM IS GIVEN.")
            exit()

    def train_model_by_reinforce(self, probs, rewards):
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

    def train_model_by_actor_critic(self, ac_losses):
        # self.optimiser_pi.zero_grad()
        # for loss in losses:
        #     loss.mean().backward()
        #     # make_dot(loss.mean(), params=dict(model.named_parameters())).render(
        #     #   "graph", format="png")
        # self.optimiser_pi.step()

        loss_len = len(ac_losses)
        loss = torch.cat(ac_losses).sum()
        loss = loss/loss_len
        self.optimiser_pi.zero_grad()
        loss.backward()
        self.optimiser_pi.step()

    def train_model_by_ppo(self, model, histories, histories_prime, states, probs, rewards):
        steps = len(histories)
        batches = len(histories[0])
        probs = torch.stack(probs).detach()
        probs = torch.reshape(probs, (probs.shape[0] * probs.shape[1], probs.shape[2]))

        # reducing dimension for parallel calculation
        t_states = torch.stack(histories)
        t_states = torch.reshape(t_states,
                                 (t_states.shape[0] * t_states.shape[1], t_states.shape[2], t_states.shape[3]))
        t_states_prime = torch.stack(histories_prime)
        t_states_prime = torch.reshape(t_states_prime, (t_states_prime.shape[0] * t_states_prime.shape[1], t_states_prime.shape[2], t_states_prime.shape[3]))
        t_rewards = torch.stack(rewards)
        t_rewards = torch.reshape(t_rewards, (t_rewards.shape[0] * t_rewards.shape[1], 1))
        t_actions = torch.stack(states)
        t_actions = torch.reshape(t_actions, (t_actions.shape[0] * t_actions.shape[1], t_actions.shape[2]))

        for _ in range(self.ppo_iter):
            # calculating advantage
            td_target = t_rewards + self.gamma * model.v(t_states_prime)
            v = model.v(t_states)
            delta = td_target - v
            delta = torch.reshape(delta, (steps, batches, 1)).detach()
            deltas = [delta[i] for i in range(len(delta))]

            advantage_list = []
            advantage = torch.zeros((len(deltas[0]), 1), device=self.device)
            for delta in deltas[::-1]:
                advantage = delta + self.gamma * self.lmbda * advantage
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantage_list = torch.stack(advantage_list)
            advantage_list = torch.reshape(advantage_list, (advantage_list.shape[0] * advantage_list.shape[1], advantage_list.shape[2]))


            # calculating action probability ratio
            cur_distribution = model.get_distribution(t_states)
            cur_probs = cur_distribution.log_prob(t_actions)
            ratio = torch.exp(cur_probs - probs)

            #print("advantage:", advantage_list.mean())
            surr1 = ratio * advantage_list
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_list
            loss_clip = -torch.min(surr1, surr2)
            loss_value = F.smooth_l1_loss(td_target.detach(), v) #td_target.detach()로 수정함 220404
            loss = loss_clip.mean() #+ loss_value #+ torch.min(loss_clip) + torch.min(loss_value)

            # print("policy loss: ", loss.item(), "=", loss_clip.mean().item(), "+", loss_value.item())

            if torch.isnan(loss).any():
                print(loss)
                print("WARNING: NaN LOSS")
                None

            #print("policy loss", loss.mean())
            self.optimiser_pi.zero_grad()
            loss.backward()
            self.optimiser_pi.step()

    def collect_trajectory(self, model, tr_dataset):
        tr_x = tr_dataset.tr_tensor_x
        tr_y = tr_dataset.tr_tensor_y
        tr_x_init = tr_dataset.tr_tensor_x_init
        tr_c_init = tr_dataset.tr_tensor_c_init

        va_log = tr_dataset.va_tensor_log

        fot_duration = va_log.shape[1]
        history_length = tr_x.shape[1]
        simulation_duration = fot_duration - history_length

        history = tr_x_init
        histories = []
        histories_prime = []
        states = []
        state_log_probs = []
        rewards = []
        ac_losses = []
        self.sut.reset()
        for i in range(simulation_duration):
            cur_v = model.v(history)

            # env model state transition
            histories.append(history)
            state_dist = model.get_distribution(history)
            selected_state = state_dist.sample().detach()
            states.append(selected_state)
            log_prob = state_dist.log_prob(selected_state)
            state_log_probs.append(log_prob)

            # reward calculation
            reward = self.get_reward(history, selected_state).detach()
            rewards.append(reward)

            # cps action selection
            action = self.sut.act_parallel(history, selected_state, tr_c_init)
            # sys_operations = torch.tensor(sys_operations).to(device=self.device).type(torch.float32)
            next_step = torch.cat((selected_state, action), dim=1)
            next_step = torch.reshape(next_step, (next_step.shape[0], 1, next_step.shape[1]))
            history = history[:, 1:]
            history = torch.cat((history, next_step), dim=1)
            histories_prime.append(history)

            # calculate actor-critic loss (this code is only for conviniant calculation of actor-critic loss)
            next_v = model.v(history)
            td_target = reward + self.gamma * next_v
            delta = td_target - cur_v
            ac_loss = -log_prob * delta.detach() + delta * delta
            ac_losses.append(ac_loss)

        return histories, histories_prime, states, state_log_probs, rewards, ac_losses

    def get_reward(self, history, state):
        reward = self.discriminator.forward(history, state)
        reward = -reward.log()
        reward = torch.nan_to_num(reward)
        #reward = torch.tanh(reward)
        return reward.detach()

    def update_discriminator(self, model, tr_dataset):
        model.eval()
        self.discriminator.train()

        tr_x = tr_dataset.tr_tensor_x
        tr_y = tr_dataset.tr_tensor_y
        tr_x_init = tr_dataset.tr_tensor_x_init
        tr_c_init = tr_dataset.tr_tensor_c_init

        va_log = tr_dataset.va_tensor_log

        num_state_features = tr_y.shape[1]
        fot_duration = va_log.shape[1]
        history_length = tr_x.shape[1]
        simulation_duration = fot_duration - history_length

        # simulate
        sim_log = non_deterministic_simulation(self.sut, model, simulation_duration, tr_c_init, tr_x_init)

        # transform simulation log
        sim_x, sim_y = tensor_log_to_tensor_xys(sim_log, simulation_duration, history_length, num_state_features)

        # train discriminator
        self.supervised_learning_discriminator(tr_x, tr_y, sim_x, sim_y)

    def supervised_learning_discriminator(self, exp_history, exp_state, model_history, model_state):
        exp_label = torch.zeros(exp_history.shape[0], device=self.device)
        model_label = torch.ones(model_history.shape[0], device=self.device)

        histories = torch.cat((exp_history, model_history), dim=0)
        states = torch.cat((exp_state, model_state), dim=0)
        labels = torch.cat((exp_label, model_label), dim=0)
        labels = torch.reshape(labels, (labels.shape[0], 1))

        disc_tr_dl = DataLoader(dataset=TensorDataset(histories, states, labels), batch_size=512, shuffle=True)

        for epoch in range(self.disc_iter):
            # losses = []
            for _, (history_batch, state_batch, label_batch) in enumerate(disc_tr_dl):
                judge = self.discriminator(history_batch, state_batch)
                loss = self.disc_loss(judge, label_batch)
                # losses.append(loss.item())

                self.optimiser_d.zero_grad()
                loss.backward()
                self.optimiser_d.step()
            # mean_loss = np.mean(losses)
            # print("Disc loss:", mean_loss)



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
