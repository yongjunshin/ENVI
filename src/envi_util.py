import copy

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from src.simulation import non_deterministic_simulation, get_dtw_similarity, get_euclidean_similarity, \
    deterministic_simulation


def load_best_model(legacy_model, model_dir, model_prefix, losses, epoches):
    best_model = copy.deepcopy(legacy_model)

    loss_min_idx = losses.index(min(losses))
    loss_min_epoch = epoches[loss_min_idx]
    print("best model idx:", loss_min_epoch)
    best_model_file_name = model_dir + model_prefix + "_" + str(loss_min_epoch) + ".pt"

    best_model.load_state_dict(torch.load(best_model_file_name))
    return best_model


def save_model(model, model_dir_path, model_prefix, model_epch):
    file_name = model_dir_path + model_prefix + "_" + str(model_epch) + ".pt"
    torch.save(model.state_dict(), file_name)


def save_best_model(model, model_dir_name):
    torch.save(model.state_dict(), model_dir_name)


def evaluate_model(self, model, deterministic_flag, tr_dataset, envi_algo):
    model.eval()

    tr_x = tr_dataset.tr_tensor_x
    tr_y = tr_dataset.tr_tensor_y
    tr_x_init = tr_dataset.tr_tensor_x_init
    tr_c_init = tr_dataset.tr_tensor_c_init
    tr_log = tr_dataset.tr_tensor_log

    va_x = tr_dataset.va_tensor_x
    va_y = tr_dataset.va_tensor_y
    va_x_init = tr_dataset.va_tensor_x_init
    va_c_init = tr_dataset.va_tensor_c_init
    va_log = tr_dataset.va_tensor_log

    num_state_features = tr_y.shape[1]
    fot_duration = va_log.shape[1]
    history_length = tr_x.shape[1]
    simulation_duration = fot_duration - history_length

    # model loss evaluation

    tr_loss = evaluate_model_loss(model, (tr_x, tr_y), deterministic_flag)
    va_loss = evaluate_model_loss(model, (va_x, va_y), deterministic_flag)

    # model simulation
    if deterministic_flag:
        tr_simul_log = deterministic_simulation(self.sut, model, simulation_duration, tr_c_init, tr_x_init)
        va_simul_log = deterministic_simulation(self.sut, model, simulation_duration, va_c_init, va_x_init)
    else:
        tr_simul_log = non_deterministic_simulation(self.sut, model, simulation_duration, tr_c_init, tr_x_init)
        va_simul_log = non_deterministic_simulation(self.sut, model, simulation_duration, va_c_init, va_x_init)

    # model euclidean loss evaluation
    tr_euclidean_loss = get_euclidean_similarity(tr_simul_log, tr_log, num_state_features)
    va_euclidean_loss = get_euclidean_similarity(va_simul_log, va_log, num_state_features)

    # model dtw loss evaluation
    tr_dtw_loss = get_dtw_similarity(tr_simul_log, tr_log, num_state_features)
    va_dtw_loss = get_dtw_similarity(va_simul_log, va_log, num_state_features)

    return tr_loss, va_loss, tr_euclidean_loss, va_euclidean_loss, tr_dtw_loss, va_dtw_loss


def evaluate_model_loss(model, eval_data, deterministic_flag):
    model.eval()
    loss_fn = torch.nn.MSELoss()
    x = eval_data[0]
    y = eval_data[1]
    dl = DataLoader(dataset=TensorDataset(x, y), batch_size=512, shuffle=True)

    loss_acum = 0
    iter = 0
    for i, (x_batch, y_batch) in enumerate(dl):
        if deterministic_flag:
            # deterministic
            y_pred = model(x_batch)
            loss = loss_fn(y_batch, y_pred)
            loss_acum = loss_acum + loss.item()
        else:
            # non-deterministic
            bc_dist = model.get_distribution(x_batch)
            loss = -bc_dist.log_prob(y_batch)
            loss = loss.mean()
            loss_acum = loss_acum + loss.item()
        iter = iter + 1

    return loss_acum / iter


def draw_plot(title, data_list, name_list, x_axis=None, fig_size=(9,3)):
    plt.figure(figsize=fig_size)
    default_x_ticks = range(len(data_list[0]))
    for idx in range(len(data_list)):
        plt.plot(data_list[idx], label=name_list[idx])
    if x_axis != None:
        plt.xticks(default_x_ticks, x_axis)
    plt.legend()
    plt.title(title)
    plt.show()
