import glob
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random


def load_tmvk_norm_np_data(fot_dir, tr_configs, num_tr_fots, num_va_fots, num_te_fots, remove_noise):
    # read FOT logs and append them in a list
    raw_dfs = []
    num_fots = []
    min_rows = None
    for config in tr_configs:
        file_name_template = fot_dir + "log_" + str(config) + "_*.csv"
        file_names = glob.glob(file_name_template)
        num_fots.append(len(file_names))
        for file in file_names:
            raw_df = pd.read_csv(file)
            raw_df = raw_df[['color', 'angle']]
            if min_rows == None:
                min_rows = len(raw_df)
            else:
                if len(raw_df) < min_rows:
                    min_rows = len(raw_df)
            raw_dfs.append(raw_df)

    # normalize FOT logs together
    norm_nps, scaler = normalize_dataframes_to_nparrays(raw_dfs, remove_noise=remove_noise)

    # split normalized FOT logs into independent FOTs
    norm_nps_list = []
    prev_index = 0
    for num in num_fots:
        norm_nps_list.append(norm_nps[prev_index: prev_index + num])
        prev_index = prev_index + num

    # cut log tails
    for config_idx in range(len(norm_nps_list)):
        for trial_idx in range(len(norm_nps_list[config_idx])):
            norm_nps_list[config_idx][trial_idx] = norm_nps_list[config_idx][trial_idx][:min_rows]

    # split FOT logs for each purpose, i.e., training, validation, testing
    tr_fot_data_np_list = []
    va_fot_data_np_list = []
    te_fot_data_np_list = []
    for fot_repeat_list in norm_nps_list:
        random.shuffle(fot_repeat_list)
        tr_fot_data_np_list.append(fot_repeat_list[:num_tr_fots])
        va_fot_data_np_list.append(fot_repeat_list[num_tr_fots:num_tr_fots+num_va_fots])
        te_fot_data_np_list.append(fot_repeat_list[num_tr_fots+num_va_fots:num_tr_fots+num_va_fots+num_te_fots])

    return tr_fot_data_np_list, va_fot_data_np_list, te_fot_data_np_list, scaler


def load_tmvu_norm_np_data(fot_dir, tr_configs, te_configs, num_tr_fots, num_va_fots, num_te_fots, remove_noise):
    # read FOT logs and append them in a list
    raw_dfs = []
    num_fots = []
    min_rows = None
    for config in tr_configs + te_configs:
        file_name_template = fot_dir + "log_" + str(config) + "_*.csv"
        file_names = glob.glob(file_name_template)
        num_fots.append(len(file_names))
        for file in file_names:
            raw_df = pd.read_csv(file)
            raw_df = raw_df[['color', 'angle']]
            if min_rows == None:
                min_rows = len(raw_df)
            else:
                if len(raw_df) < min_rows:
                    min_rows = len(raw_df)
            raw_dfs.append(raw_df)

    # normalize FOT logs together
    norm_nps, scaler = normalize_dataframes_to_nparrays(raw_dfs, remove_noise=remove_noise)

    # split normalized FOT logs into independent FOTs
    norm_nps_list = []
    prev_index = 0
    for num in num_fots:
        norm_nps_list.append(norm_nps[prev_index: prev_index + num])
        prev_index = prev_index + num

    # cut log tails
    for config_idx in range(len(norm_nps_list)):
        for trial_idx in range(len(norm_nps_list[config_idx])):
            norm_nps_list[config_idx][trial_idx] = norm_nps_list[config_idx][trial_idx][:min_rows]

    tr_norm_nps_list = norm_nps_list[:len(tr_configs)]
    te_norm_nps_list = norm_nps_list[len(tr_configs):]

    # split FOT logs for each purpose, i.e., training, validation, testing
    tr_fot_data_np_list = []
    va_fot_data_np_list = []
    te_fot_data_np_list = []
    for fot_repeat_list in tr_norm_nps_list:
        random.shuffle(fot_repeat_list)
        tr_fot_data_np_list.append(fot_repeat_list[:num_tr_fots])
        va_fot_data_np_list.append(fot_repeat_list[num_tr_fots:num_tr_fots + num_va_fots])

    for fot_repeat_list in te_norm_nps_list:
        te_fot_data_np_list.append(fot_repeat_list[:num_te_fots])

    return tr_fot_data_np_list, va_fot_data_np_list, te_fot_data_np_list, scaler


def normalize_dataframes_to_nparrays(raw_dfs: list, remove_noise=False) -> (list, MinMaxScaler):
    """
    Normalize list of dataframes and return normalzed list of nparrays

    :param raw_dfs: list of pd.DataFrames (list)
    :return list of np.ndarrays (list)
    :return used MinMaxScaler (MinMaxScaler)
    """

    raw_nps = [df.to_numpy() for df in raw_dfs]
    if remove_noise:
        raw_nps = remove_black_noise(raw_nps)

    # Merge files
    # data_shape_list_for_each_file = [raw_df.shape for raw_df in raw_dfs]
    # raw_concat_df = pd.concat(raw_dfs)
    data_shape_list_for_each_file = [raw_np.shape for raw_np in raw_nps]
    raw_concat_np = np.concatenate(raw_nps)

    # Normalize data
    mm = MinMaxScaler(feature_range=(-1, 1))
    normalized_concat_nparray = mm.fit_transform(raw_concat_np)

    # Split files
    noramlized_nparrays = []
    prev_index = 0
    for shape in data_shape_list_for_each_file:
        noramlized_nparrays.append(normalized_concat_nparray[prev_index: prev_index + shape[0], :])
        prev_index = prev_index + shape[0]

    return noramlized_nparrays, mm


def np_log_to_tensor_training_data(config_log_list, config_list, sliding_window_size, num_state_features, device):
    initial_x = []
    x_datapoints = []
    y_datapoints = []
    initial_configs = []
    config_idx = 0
    full_configs = []
    for log_list in config_log_list:
        for log in log_list:
            i = 0
            initial_x.append(log[i:i+sliding_window_size])
            initial_configs.append(config_list[config_idx])
            while i < len(log)-sliding_window_size:
                x_datapoints.append(log[i:i+sliding_window_size])
                y_datapoints.append(log[i+sliding_window_size][:num_state_features])
                full_configs.append(config_list[config_idx])
                i = i + 1
        config_idx = config_idx + 1
    x_datapoints = torch.tensor(np.array(x_datapoints), dtype=torch.float32, device=device)
    y_datapoints = torch.tensor(np.array(y_datapoints), dtype=torch.float32, device=device)
    full_configs = torch.tensor(np.array(full_configs), dtype=torch.float32, device=device)
    initial_x = torch.tensor(np.array(initial_x), dtype=torch.float32, device=device)
    initial_configs = torch.tensor(np.array(initial_configs), dtype=torch.float32, device=device)
    flat_fot_logs = torch.tensor(np.concatenate([np.stack(log_list) for log_list in config_log_list]), dtype=torch.float32, device=device)

    return x_datapoints, y_datapoints, full_configs, initial_x, initial_configs, flat_fot_logs


def remove_black_noise(raw_np_list):
    black_noise_upper_bound = 1000
    black_noise_lower_bound = 10

    clean_np_list = []
    for raw_np in raw_np_list:
        distances = raw_np[:, 1]
        distance_noise_flag_1 = np.where(distances >= black_noise_upper_bound, 1, 0)
        distance_noise_flag_2 = np.where(distances <= black_noise_lower_bound, 1, 0)
        distance_noise_flag = distance_noise_flag_1 + distance_noise_flag_2
        num_distance_black_noise = np.sum(distance_noise_flag)
        if num_distance_black_noise == 0:
            clean_np_list.append(raw_np)
        else:
            pre_dis = distances[0]
            for idx in range(1, len(distances)):
                if distances[idx] >= black_noise_upper_bound or distances[idx] <= black_noise_lower_bound:
                    distances[idx] = pre_dis
                else:
                    pre_dis = distances[idx]
            pre_dis = distances[-1]
            for idx in reversed(range(0, len(distances)-1)):
                if distances[idx] >= black_noise_upper_bound or distances[idx] <= black_noise_lower_bound:
                    distances[idx] = pre_dis
                else:
                    pre_dis = distances[idx]
            raw_np[:, 1] = distances
            clean_np_list.append(raw_np)
    return clean_np_list