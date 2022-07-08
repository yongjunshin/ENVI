import glob
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

from src.data_manager_av_fot import remove_black_noise


def load_all_csv_log_data(fot_dir):
    raw_nps = []
    file_name_template = fot_dir + "*_140_*.csv"
    file_names = glob.glob(file_name_template)
    for file in file_names:
        raw_df = pd.read_csv(file)
        raw_df = raw_df[[' color', ' distance', ' angle', ' speed']]
        raw_nps.append(raw_df.to_numpy())
    return raw_nps





FOT_LOG_DIR = "data/AV-FOT/"
logs = load_all_csv_log_data(FOT_LOG_DIR)
num_logs = len(logs)
print("num logs:", num_logs, "logs")

min_len = logs[0].shape[0]
max_len = 0
for log in logs:
    log_len = log.shape[0]
    if min_len > log_len:
        min_len = log_len
    if max_len < log_len:
        max_len = log_len

print("min FOT log length:", min_len, "tick")
print("max FOT log length:", max_len, "tick")


# remove black noise
logs = remove_black_noise(logs)


num_log_with_black_noise_color = 0
num_log_with_black_noise_distance = 0
max_num_noise_distance = None
idx = 0
color_min = np.min(logs[0][:, 0])
color_max = np.max(logs[0][:, 0])
dist_min = np.min(logs[0][:, 1])
dist_max = np.max(logs[0][:, 1])


black_noise_upper_bound_color = 100
black_noise_lower_bound_color = 0
black_noise_upper_bound_distance = 2550
black_noise_lower_bound_distance = 10

for log in logs:
    colors = log[:, 0]
    colors_noise_flag_1 = np.where(colors >= black_noise_upper_bound_color, 1, 0)
    colors_noise_flag_2 = np.where(colors <= black_noise_lower_bound_color, 1, 0)
    colors_noise_flag = colors_noise_flag_1 + colors_noise_flag_2
    num_color_black_noise = np.sum(colors_noise_flag)
    if num_color_black_noise > 0:
        num_log_with_black_noise_color = num_log_with_black_noise_color + 1

    distances = log[:, 1]
    distance_noise_flag_1 = np.where(distances >= black_noise_upper_bound_distance, 1, 0)
    distance_noise_flag_2 = np.where(distances <= black_noise_lower_bound_distance, 1, 0)
    distance_noise_flag = distance_noise_flag_1 + distance_noise_flag_2
    num_distance_black_noise = np.sum(distance_noise_flag)
    if num_distance_black_noise > 0:
        num_log_with_black_noise_distance = num_log_with_black_noise_distance + 1
        if max_num_noise_distance == None:
            max_num_noise_distance = num_distance_black_noise
            print(idx)
        else:
            if max_num_noise_distance < num_distance_black_noise:
                max_num_noise_distance = num_distance_black_noise
                print(idx)

    if color_min > np.min(colors):
        color_min = np.min(colors)
    if color_max < np.max(colors):
        color_max = np.max(colors)
    if dist_min > np.min(distances):
        dist_min = np.min(distances)
    if dist_max < np.max(distances):
        dist_max = np.max(distances)

    idx = idx + 1

print("num logs with black noise color:", num_log_with_black_noise_color, "logs")
print("num logs with black noise distance:", num_log_with_black_noise_distance, "logs")
print("max num noise in distance", max_num_noise_distance)

print("min color: ", color_min)
print("max color: ", color_max)
print("min dist: ", dist_min)
print("max dist: ", dist_max)






