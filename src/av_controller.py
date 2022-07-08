import numpy as np
import torch


class AV:
    def __init__(self, normalizer):
        self.normalizer = normalizer
        self.activate = False
        self.color_goal = 33
        self.distance_goal = 200


        self.lk_p = None
        self.lk_i = 0.1
        self.lk_d = 0.5
        self.cc_p = None
        self.cc_i = 0.0
        self.cc_d = 0.3

        self.color_deviation = None
        self.color_derivative = None
        self.color_integral = None
        self.color_last_deviation = None

        self.distance_deviation = None
        self.distance_derivative = None
        self.distance_integral = None
        self.distance_last_deviation = None

    def reset(self):
        self.activate = False

        self.color_deviation = None
        self.color_derivative = None
        self.color_integral = None
        self.color_last_deviation = None

        self.distance_deviation = None
        self.distance_derivative = None
        self.distance_integral = None
        self.distance_last_deviation = None

    def act_parallel(self, history, state, configs):
        np_history = history.cpu().numpy()
        np_state = state.cpu().numpy()
        np_configs = configs.cpu().numpy()

        num_test = np_history.shape[0]
        history_length = np_history.shape[1]
        num_history_features = np_history.shape[2]
        num_state_features = np_state.shape[1]
        num_action_features = num_history_features - num_state_features
        self.lk_p = np_configs[:, 0]
        self.cc_p = np_configs[:, 1]

        reshaped_history = np.reshape(np_history, (num_test * history_length, num_history_features))
        denorm_history = self.normalizer.inverse_transform(reshaped_history)
        denorm_history = np.reshape(denorm_history, np_history.shape)
        reshaped_state = np.concatenate((np_state, np.zeros((num_test, num_action_features))), axis=1)
        denorm_state = self.normalizer.inverse_transform(reshaped_state)
        denorm_state = np.reshape(denorm_state[:, :num_state_features], np_state.shape)

        if self.activate == False:
            self.color_deviation = np.zeros(num_test)
            self.color_derivative = np.zeros(num_test)
            self.color_integral = np.zeros(num_test)
            self.color_last_deviation = np.zeros(num_test)

            self.distance_deviation = np.zeros(num_test)
            self.distance_derivative = np.zeros(num_test)
            self.distance_integral = np.zeros(num_test)
            self.distance_last_deviation = np.zeros(num_test)

            for i in range(np_history.shape[1]):
                color = denorm_history[:, i, 0]
                distance = denorm_history[:, i, 1]
                turn_rate = self.__run_lk_pid(color)
                drive_speed = self.__run_cc_pid(distance)

            color = denorm_state[:, 0]
            distance = denorm_state[:, 1]
            turn_rate = self.__run_lk_pid(color)
            drive_speed = self.__run_cc_pid(distance)

            self.activate = True
            turn_rate = np.reshape(turn_rate, (num_test, 1))
            drive_speed = np.reshape(drive_speed, (num_test, 1))

            action = np.concatenate([np.zeros(np_state.shape), turn_rate, drive_speed], axis=1)
            norm_action = self.normalizer.transform(action)
            norm_action = norm_action[:, num_state_features:]
            norm_action = torch.tensor(norm_action, dtype=torch.float32, device=history.get_device())
            return norm_action
        else:
            color = denorm_state[:, 0]
            distance = denorm_state[:, 1]
            turn_rate = self.__run_lk_pid(color)
            drive_speed = self.__run_cc_pid(distance)

            turn_rate = np.reshape(turn_rate, (num_test, 1))
            drive_speed = np.reshape(drive_speed, (num_test, 1))

            action = np.concatenate([np.zeros(np_state.shape), turn_rate, drive_speed], axis=1)
            norm_action = self.normalizer.transform(action)
            norm_action = norm_action[:, num_state_features:]
            norm_action = torch.tensor(norm_action, dtype=torch.float32, device=history.get_device())
            return norm_action

    def get_normalizer(self):
        return self.normalizer

    def __run_lk_pid(self, color_observation):
        self.color_deviation = color_observation - self.color_goal
        self.color_integral = self.color_integral + self.color_deviation
        self.color_derivative = self.color_deviation - self.color_last_deviation
        turn_rate = (self.lk_p * self.color_deviation) + (self.lk_i * self.color_integral) + (self.lk_d * self.color_derivative)
        self.color_last_deviation = self.color_deviation
        return turn_rate

    def __run_cc_pid(self, distance_observation):
        self.distance_deviation = distance_observation - self.distance_goal
        self.distance_integral = self.distance_integral + self.distance_deviation
        self.distance_derivative = self.distance_deviation - self.distance_last_deviation
        drive_speed = (self.cc_p * self.distance_deviation) + (self.cc_i * self.distance_integral) + (self.cc_d * self.distance_derivative)
        self.distance_last_deviation = self.distance_deviation
        return drive_speed
