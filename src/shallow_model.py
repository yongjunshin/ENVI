import torch
import torch.nn as nn

class LineTracerShallowRFEnvironmentModel(nn.Module):
    def __init__(self, regressor, num_state_features, device):
        super(LineTracerShallowRFEnvironmentModel, self).__init__()
        self.regressor = regressor
        self.device = device
        self.num_state_features = num_state_features

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """
        batch_size = x.shape[0]

        flatted_x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).cpu()

        predicted_y = self.regressor.predict(flatted_x)
        predicted_y = torch.tensor(predicted_y, device=self.device)
        predicted_y = torch.reshape(predicted_y, (x.shape[0], self.num_state_features))

        return predicted_y



class LineTracerShallowPREnvironmentModel(nn.Module):
    def __init__(self, regressor, poly_feature, num_state_features, device):
        super(LineTracerShallowPREnvironmentModel, self).__init__()
        self.regressor = regressor
        self.poly_feature = poly_feature
        self.num_state_features = num_state_features
        self.device = device

    def forward(self, x):
        """
        Deterministic (mean) action
        :param x: state
        :return: action
        """
        batch_size = x.shape[0]

        flatted_x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])).cpu()
        poly_flatted_x = self.poly_feature.transform(flatted_x)

        predicted_y = self.regressor.predict(poly_flatted_x)
        predicted_y = torch.tensor(predicted_y, device=self.device)
        predicted_y = torch.reshape(predicted_y, (batch_size, self.num_state_features))
        predicted_y = torch.nan_to_num(predicted_y)
        predicted_y = torch.clamp(predicted_y, -1, 1)

        return predicted_y
