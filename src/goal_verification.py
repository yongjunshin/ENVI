import torch
import numpy as np


def get_maximum_displacements(trajectories, feature_idx, denorm_goal, scaler, sim_start_idx=0):
    denorm_features = []
    for i in range(trajectories.shape[2]):
        if i == feature_idx:
            denorm_features.append(denorm_goal)
        else:
            denorm_features.append(0)
    denorm_features = np.array(denorm_features)
    denorm_features = np.reshape(denorm_features, (1, denorm_features.shape[0]))
    norm_features = scaler.transform(denorm_features)
    norm_goal = norm_features[0, feature_idx]

    feature_traj = trajectories[:, sim_start_idx:, [feature_idx]]
    displacements = feature_traj - norm_goal
    displacements = torch.abs(displacements)
    maximum_displacements = torch.amax(displacements, dim=1)
    return maximum_displacements


def get_minimum_front_distance(trajectories, feature_idx, denorm_safe_distance, scaler, sim_start_idx=0):
    denorm_features = []
    for i in range(trajectories.shape[2]):
        if i == feature_idx:
            denorm_features.append(denorm_safe_distance)
        else:
            denorm_features.append(0)
    denorm_features = np.array(denorm_features)
    denorm_features = np.reshape(denorm_features, (1, denorm_features.shape[0]))
    norm_features = scaler.transform(denorm_features)
    norm_goal = norm_features[0, feature_idx]

    feature_traj = trajectories[:, sim_start_idx:, [feature_idx]]
    displacements = feature_traj - norm_goal
    # displacements = torch.abs(displacements)
    minimum_front_distances = torch.amin(displacements, dim=1)
    return minimum_front_distances


def get_maximum_jerk(trajectories, feature_idx, sim_start_idx=0):
    feature_traj = trajectories[:, sim_start_idx:, [feature_idx]]
    dev1 = feature_traj[:, 1:] - feature_traj[:, :-1]
    dev2 = dev1[:, 1:] - dev1[:, :-1]
    dev3 = dev2[:, 1:] - dev2[:, :-1]

    dev3 = torch.abs(dev3)
    maximum_jerks = torch.amax(dev3, dim=1)
    return maximum_jerks


def get_average_jerk(trajectories, feature_idx, sim_start_idx=0):
    feature_traj = trajectories[:, sim_start_idx:, [feature_idx]]
    dev1 = feature_traj[:, 1:] - feature_traj[:, :-1]
    dev2 = dev1[:, 1:] - dev1[:, :-1]
    dev3 = dev2[:, 1:] - dev2[:, :-1]

    dev3 = torch.abs(dev3)
    maximum_jerks = torch.mean(dev3, dim=1)
    return maximum_jerks


