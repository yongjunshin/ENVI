import torch
import torch.nn as nn
from torch.distributions import Normal


class AVEnvModelDNN(nn.Module):
    def __init__(self, input_features, history_len, hidden_fc_features, output_features, device):
        super(AVEnvModelDNN, self).__init__()
        self.hidden_dim = hidden_fc_features

        conv_out_chan = 16
        conv_k = 3
        pool_k = 2
        self.conv1d = nn.Conv1d(in_channels=input_features, out_channels=conv_out_chan, kernel_size=conv_k)
        conv_l_out = history_len - conv_k + 1
        self.max_pool1d = nn.MaxPool1d(kernel_size=pool_k)
        pool_l_out = int((conv_l_out - pool_k)/pool_k + 1)
        self.fc_in = nn.Linear(conv_out_chan * pool_l_out, hidden_fc_features)
        self.actor_fc_middle_1 = nn.Linear(hidden_fc_features, hidden_fc_features)
        # self.actor_fc_middle_2 = nn.Linear(hidden_fc_features, hidden_fc_features)
        # self.actor_fc_middle_3 = nn.Linear(hidden_fc_features, hidden_fc_features)
        self.actor_fc_out = nn.Linear(hidden_fc_features, output_features)
        self.actor_fc_std = nn.Linear(hidden_fc_features, output_features)

        self.critic_fc_middle_1 = nn.Linear(hidden_fc_features, hidden_fc_features)
        # self.critic_fc_middle_2 = nn.Linear(hidden_fc_features, hidden_fc_features)
        # self.critic_fc_middle_3 = nn.Linear(hidden_fc_features, hidden_fc_features)
        self.critic_fc_v = nn.Linear(hidden_fc_features, 1)

        self.device = device
        self.to(device)

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """
        #x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
        x = torch.swapaxes(x, 1, 2)
        out = torch.relu(torch.nan_to_num(self.conv1d(x)))
        out = self.max_pool1d(out)
        out = torch.flatten(out, start_dim=1)
        out = torch.relu(torch.nan_to_num(self.fc_in(out)))
        out = torch.relu(torch.nan_to_num(self.actor_fc_middle_1(out)))
        # out = torch.relu(torch.nan_to_num(self.actor_fc_middle_2(out)))
        # out = torch.relu(torch.nan_to_num(self.actor_fc_middle_3(out)))
        out = torch.tanh(torch.nan_to_num(self.actor_fc_out(out)))
        return out

    def get_distribution(self, x):
        """
        Distribution of non-deterministic Normal(mean, std) action
        :param x: state
        :return: distribution
        """
        x = torch.swapaxes(x, 1, 2)
        out = torch.relu(torch.nan_to_num(self.conv1d(x)))
        out = self.max_pool1d(out)
        out = torch.flatten(out, start_dim=1)
        out = torch.relu(torch.nan_to_num(self.fc_in(out)))
        out = torch.relu(torch.nan_to_num(self.actor_fc_middle_1(out)))
        # out = torch.relu(torch.nan_to_num(self.actor_fc_middle_2(out)))
        # out = torch.relu(torch.nan_to_num(self.actor_fc_middle_3(out)))
        mu = torch.tanh(torch.nan_to_num(self.actor_fc_out(out)))

        sigma = torch.sigmoid(torch.nan_to_num(self.actor_fc_std(out)))# + torch.finfo(torch.float32).eps

        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            print(out)
            print(mu)
            print(sigma)

        if torch.numel(sigma) - torch.count_nonzero(sigma) > 0:
            sigma = sigma + torch.finfo(torch.float32).eps

        dist = Normal(mu, sigma)
        return dist

    def act(self, x):
        """
        sampled non-deterministic action
        :param x: state
        :return: action
        """
        dist = self.get_distribution(x)
        action = dist.sample()
        action = action.detach()
        return action

    def v(self, x):
        """
        value (sum discounted reward) estimation function
        :param x: state
        :return: value
        """
        x = torch.swapaxes(x, 1, 2)
        out = torch.relu(torch.nan_to_num(self.conv1d(x)))
        out = self.max_pool1d(out)
        out = torch.flatten(out, start_dim=1)
        out = torch.relu(torch.nan_to_num(self.fc_in(out)))
        out = torch.relu(torch.nan_to_num(self.critic_fc_middle_1(out)))
        # out = torch.relu(torch.nan_to_num(self.critic_fc_middle_2(out)))
        # out = torch.relu(torch.nan_to_num(self.critic_fc_middle_3(out)))
        v = torch.nan_to_num(self.critic_fc_v(out))
        return v
