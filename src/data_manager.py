import glob
import math

import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

def collect_config_csvs_file_names_from_dir(dir, config_string):
    # return a list of FOT log file names of the given CPS configuration (string).
    file_name_template = dir + "log_" + config_string + "_*.csv"
    file_names = glob.glob(file_name_template)
    return file_names


def read_columns_from_csv_files(file_names, columns):
    # read columns in the csv files and return them as a list of nparrays.
    np_list = []
    for file in file_names:
        raw_df = pd.read_csv(file)
        raw_df = raw_df[columns]
        np_list.append(raw_df.to_numpy())
    return np_list

def normalize_multiple_nparray(np_list):
    # normalize multiple nparrays together and return a list of normalized nparryas and a shared scaler
    original_shape_list = [np.shape for np in np_list]
    concat_np_list = np.concatenate(np_list)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    norm_concat_np_list = scaler.fit_transform(concat_np_list)

    norm_np_list = []
    prev_index = 0
    for shape in original_shape_list:
        norm_np_list.append(norm_concat_np_list[prev_index: prev_index + shape[0]])
        prev_index = prev_index + shape[0]

    return norm_np_list, scaler


def cut_tailes_of_logs(np_list):
    original_len_list = [np.shape[0] for np in np_list]
    min_len = min(original_len_list)

    tail_cut_np_list = []
    for nparr in np_list:
        tail_cut_np_list.append(nparr[:min_len])

    return tail_cut_np_list


def an_np_fot_log_to_tensor_xyc_data(np_array, history_length, state_features, config, device):
    # get an np_array and return X and Y
    log_len = len(np_array)
    i = 0
    x_datapoints = []
    y_datapoints = []
    c_datapoints = []
    x_init_datapoints = []
    y_init_datapoints = []
    c_init_datapoints = []
    while i < log_len - history_length:
        if i == 0:
            x_init_datapoints.append(np_array[i:i+history_length])
            y_init_datapoints.append(np_array[i+history_length][:state_features])
            c_init_datapoints.append(list(config))
        x_datapoints.append(np_array[i:i+history_length])
        y_datapoints.append(np_array[i+history_length][:state_features])
        c_datapoints.append(list(config))
        i = i + 1

    x_datapoints = torch.tensor(np.array(x_datapoints), dtype=torch.float32, device=device)
    y_datapoints = torch.tensor(np.array(y_datapoints), dtype=torch.float32, device=device)
    c_datapoints = torch.tensor(np.array(c_datapoints), dtype=torch.float32, device=device)
    x_init_datapoints = torch.tensor(np.array(x_init_datapoints), dtype=torch.float32, device=device)
    y_init_datapoints = torch.tensor(np.array(y_init_datapoints), dtype=torch.float32, device=device)
    c_init_datapoints = torch.tensor(np.array(c_init_datapoints), dtype=torch.float32, device=device)

    return x_datapoints, y_datapoints, c_datapoints, x_init_datapoints, y_init_datapoints, c_init_datapoints


def kfold_split(list, split_nums, fold_size):
    # shuffle items in the list and split the items following the given ratios whose sum is 1
    # return a list of splited multiple lists
    num_data = len(list)

    tr_start_idx = 0
    va_start_idx = fold_size * 2
    te_start_idx = num_data - fold_size

    splited_lists = []
    splited_lists.append(list[tr_start_idx: tr_start_idx + split_nums[0]])
    splited_lists.append(list[va_start_idx: va_start_idx + split_nums[1]])
    splited_lists.append(list[te_start_idx: te_start_idx + split_nums[2]])

    return splited_lists


def list_shift(list):
    list.append(list.pop(0))

def get_median(data):
    mid_idx = int(len(data)/2)
    sorted_data = np.sort(data, axis=0)
    return sorted_data[mid_idx]

def remove_salt_and_pepper_noise(data, filter_size=5):
    new_data = []
    edge_length = int(filter_size / 2)
    for i in range(edge_length, len(data) - edge_length):
        new_data.append(get_median(data[i - edge_length:i - edge_length + filter_size]))
    return np.array(new_data)

def remove_noise(data_list):
    new_data_list = []
    cut_tail_length = 20
    for data in data_list:
        temp_data = data[:-cut_tail_length]
        temp_data = remove_salt_and_pepper_noise(temp_data)
        new_data_list.append(temp_data)
    return new_data_list


def prepare_np_fot_data(dir, configs, columns, split_nums, num_fold, fold_idx, remove_noise_flag=False):
    np_list_for_each_config = []
    num_fots_for_each_config = []

    # read logs
    for config in configs:
        file_names = collect_config_csvs_file_names_from_dir(dir, config)
        np_list = read_columns_from_csv_files(file_names, columns)
        np_list_for_each_config.append(np_list)
        num_fots_for_each_config.append(len(np_list))

    # merge all configurations, cut tails of the logs, and normalize the logs together
    concat_np_list = []
    for np_list in np_list_for_each_config:
        concat_np_list = concat_np_list + np_list
    concat_np_list = cut_tailes_of_logs(concat_np_list)
    if remove_noise_flag:
        concat_np_list = remove_noise(concat_np_list)
    concat_norm_np_list, scaler = normalize_multiple_nparray(concat_np_list)

    # split for each configuration
    norm_np_list_for_each_config = []
    i = 0
    for num_fots in num_fots_for_each_config:
        norm_np_list_for_each_config.append(concat_norm_np_list[i:i+num_fots])
        i = i + num_fots

    # training, validation, and testing log split
    fold_shift = int(math.ceil(len(norm_np_list_for_each_config[0]) / num_fold))
    for np_list in norm_np_list_for_each_config:
        for i in range(fold_idx):
            for s in range(fold_shift):
                list_shift(np_list)
                None

    split_norm_np_list_for_each_config = []
    for np_list in norm_np_list_for_each_config:
        split_norm_np_list_for_each_config.append(kfold_split(np_list, split_nums, fold_shift))

    return split_norm_np_list_for_each_config, scaler


def collect_np_data_and_config_for_purpose(np_dataset, all_configs, config_idxs, purpose_flag):
    # np_dataset structure: [config][purpose]
    # return nps and configurations
    nps = []
    configs = []
    for idx in config_idxs:
        selected_log = np_dataset[idx][purpose_flag]
        num_log = len(selected_log)
        nps = nps + selected_log
        configs = configs + [all_configs[idx]] * num_log

    return nps, configs


def np_to_tensor_xyc_data(np_log_list, np_c_list, history_len, state_features, device):
    x_list = []
    y_list = []
    c_list = []
    x_init_list = []
    y_init_list = []
    c_init_list = []

    for i in range(len(np_log_list)):
        x, y, c, x_init, y_init, c_init = an_np_fot_log_to_tensor_xyc_data(np_log_list[i], history_len, state_features, np_c_list[i], device)
        x_list.append(x)
        y_list.append(y)
        c_list.append(c)
        x_init_list.append(x_init)
        y_init_list.append(y_init)
        c_init_list.append(c_init)

    concat_x = torch.concat(x_list)
    concat_y = torch.concat(y_list)
    concat_c = torch.concat(c_list)
    concat_x_init = torch.concat(x_init_list)
    concat_y_init = torch.concat(y_init_list)
    concat_c_init = torch.concat(c_init_list)

    return concat_x, concat_y, concat_c, concat_x_init, concat_y_init, concat_c_init

def tensor_log_to_tensor_xys(sim_log, simulation_duration, history_length, num_state_features):
    sim_x = []
    sim_y = []
    for i in range(simulation_duration):
        sim_x.append(sim_log[:, i:i + history_length])
        sim_y.append(sim_log[:, i + history_length, :num_state_features])
    sim_x = torch.cat(sim_x, dim=0)
    sim_y = torch.cat(sim_y, dim=0)

    return sim_x, sim_y