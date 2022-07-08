import torch
from src.soft_dtw_cuda import SoftDTW


def deterministic_simulation(controller, env_model, simulation_duration, controller_configs, init_model_input):
    history = init_model_input
    simul_log = history
    history_len = init_model_input.shape[1]

    controller.reset()
    for i in range(simulation_duration):
        # deterministic simulation
        state = env_model(history).detach()

        action = controller.act_parallel(history, state, controller_configs)
        new_step = torch.cat((state, action), dim=1)
        new_step = torch.reshape(new_step, (new_step.shape[0], 1, new_step.shape[1]))
        simul_log = torch.cat((simul_log, new_step), dim=1)
        history = simul_log[:, -history_len:, :]

    return simul_log


def non_deterministic_simulation(controller, env_model, simulation_duration, controller_configs, init_model_input):
    history = init_model_input
    simul_log = history
    history_len = init_model_input.shape[1]

    controller.reset()
    for i in range(simulation_duration):
        # non-deterministic simulation
        dist = env_model.get_distribution(history)
        state = dist.sample().detach()

        action = controller.act_parallel(history, state, controller_configs)
        new_step = torch.cat((state, action), dim=1)
        new_step = torch.reshape(new_step, (new_step.shape[0], 1, new_step.shape[1]))
        simul_log = torch.cat((simul_log, new_step), dim=1)
        history = simul_log[:, -history_len:, :]

    return simul_log


sdtw = SoftDTW(use_cuda=torch.cuda.is_available(), gamma=0.1)


def get_dtw_similarity(trajectories_1, trajectories_2, num_state_features):
    return get_dtw(trajectories_1, trajectories_2, num_state_features).mean().item()


def get_dtw(trajectories_1, trajectories_2, num_state_features):
    traj_1 = trajectories_1[:, :, :num_state_features]
    traj_2 = trajectories_2[:, :, :num_state_features]
    return sdtw(traj_1, traj_2)


def get_euclidean_similarity(trajectories_1, trajectories_2, num_state_features):
    return get_euclidean(trajectories_1, trajectories_2, num_state_features).mean().item()


def get_euclidean(trajectories_1, trajectories_2, num_state_features):
    traj_1 = trajectories_1[:, :, :num_state_features]
    traj_2 = trajectories_2[:, :, :num_state_features]

    diffs = torch.pow(traj_1 - traj_2, 2)
    diffs = torch.sum(diffs, dim=(1, 2))
    diffs = torch.sqrt(diffs)
    return diffs
