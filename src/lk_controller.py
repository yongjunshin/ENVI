import numpy as np
import torch

# Virtual Line Tracer version 1
class LaneKeepingSystem:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.black = 5
        self.white = 75
        self.threshold = (self.black + self.white) / 2
        None

    def reset(self):
        None

    def act_parallel(self, history, state, configs):
        np_history = history.cpu().numpy()
        np_state = state.cpu().numpy()
        np_configs = configs.cpu().numpy()
        np_configs = np.reshape(np_configs, (np_configs.shape[0], 1))

        num_test = np_state.shape[0]
        num_history_features = np_history.shape[2]
        num_state_features = np_state.shape[1]
        num_action_features = num_history_features - num_state_features

        reshaped_state = np.concatenate((np_state, np.zeros((num_test, num_action_features))), axis=1)
        denorm_state = self.normalizer.inverse_transform(reshaped_state)
        denorm_state = denorm_state[:, :num_state_features]
        denorm_state = np.round(denorm_state)

        positive_check = np.where(denorm_state > self.threshold, 1.0, 0.0)
        negative_check = np.where(denorm_state < self.threshold, 1.0, 0.0)

        turning_rate = positive_check * np_configs + negative_check * (-np_configs)
        action = np.concatenate([np.zeros(np_state.shape), turning_rate], axis=1)
        norm_action = self.normalizer.transform(action)
        norm_action = norm_action[:, num_state_features:]
        norm_action = torch.tensor(norm_action, dtype=torch.float32, device=history.get_device())

        return norm_action

    def get_normalizer(self):
        return self.normalizer
