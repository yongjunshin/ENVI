import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.distributions import MultivariateNormal, kl_divergence

def get_multivariate_dist(feature1_samples, feature2_samples):
    dist_mean_1 = np.mean(feature1_samples)
    dist_mean_2 = np.mean(feature2_samples)
    dist_cov = np.cov(feature1_samples, feature2_samples)
    # if np.any(dist_cov == 0):
    #     dist_cov = dist_cov + np.finfo(float).eps
    if dist_cov[0][0] < 0:
        dist_cov[0][0] = 0
    if dist_cov[0][1] < 0:
        dist_cov[0][1] = 0
    if dist_cov[1][0] < 0:
        dist_cov[1][0] = 0
    if dist_cov[1][1] < 0:
        dist_cov[1][1] = 0

    if dist_cov[0][0] == 0:
        dist_cov[0][0] = dist_cov[0][0] + np.finfo(float).eps
    if dist_cov[1][1] == 0:
        dist_cov[1][1] = dist_cov[1][1] + np.finfo(float).eps
    dist = MultivariateNormal(torch.tensor([dist_mean_1, dist_mean_2]), torch.tensor(dist_cov))
    return dist


def visualize_two_dist_diff(dist1, dist2):
    plt.figure(figsize=(9, 6))

    dist1_feature1_simul_samples = []
    dist1_feature2_simul_samples = []
    for i in range(200):
        dist1_sample1 = dist1.sample()
        dist1_feature1_simul_samples.append(dist1_sample1[0].item())
        dist1_feature2_simul_samples.append(dist1_sample1[1].item())
    dist1_feature1_simul_samples = np.array(dist1_feature1_simul_samples)
    dist1_feature2_simul_samples = np.array(dist1_feature2_simul_samples)
    plt.scatter(dist1_feature1_simul_samples, dist1_feature2_simul_samples, s=60, facecolors='none', edgecolors='red', label='fot sample')

    dist2_feature1_simul_samples = []
    dist2_feature2_simul_samples = []
    for i in range(200):
        dist2_sample1 = dist2.sample()
        dist2_feature1_simul_samples.append(dist2_sample1[0].item())
        dist2_feature2_simul_samples.append(dist2_sample1[1].item())
    dist2_feature1_simul_samples = np.array(dist2_feature1_simul_samples)
    dist2_feature2_simul_samples = np.array(dist2_feature2_simul_samples)
    plt.scatter(dist2_feature1_simul_samples, dist2_feature2_simul_samples, s=60, facecolors='none', edgecolors='blue', label='simul sample')

    plt.show()


def mean_compare_two_lists(list1, list2):
    mean1 = np.mean(list1)
    mean2 = np.mean(list2)

    return abs(mean1-mean2)


def dist_compare_two_lists(list1, list2):
    mean1 = np.mean(list1)
    std1 = np.std(list1)

    mean2 = np.mean(list2)
    std2 = np.std(list2)

    return kld_gauss(mean1, std1, mean2, std2)


def kld_gauss(u1, s1, u2, s2):
  # general KL two Gaussians
  # u2, s2 often N(0,1)
  # https://stats.stackexchange.com/questions/7440/ +
  # kl-divergence-between-two-univariate-gaussians
  # log(s2/s1) + [( s1^2 + (u1-u2)^2 ) / 2*s2^2] - 0.5
  v1 = s1 * s1
  v2 = s2 * s2
  a = np.log(s2/s1)
  num = v1 + (u1 - u2)**2
  den = 2 * v2
  b = num / den
  return a + b - 0.5


def get_config_idxs(configs_arr):
    unique_configs = []
    idxs = []
    for i in range(len(configs_arr)):
        if not configs_arr[i] in unique_configs:
            unique_configs.append(configs_arr[i])
            idxs.append(np.where(configs_arr == configs_arr[i]))
    return idxs


def reshape_data_for_each_config(data, config_idxs_list):
    reshaped_data = []
    for config_idxs in config_idxs_list:
        reshaped_data.append(data[config_idxs])
    reshaped_data = np.stack(reshaped_data)
    return reshaped_data

# case_study = 'Lane-keeping'
case_study = 'AV'
if case_study == 'AV':
    fold_idx = 2
    baseline_fold_idx = 3
else:
    fold_idx = 2
    baseline_fold_idx = 3
num_tr_fot = 20

VERIF_RESULT_ROOT_PATH = "output/" + case_study + "/fold_" + str(fold_idx) + "_tr_" + str(num_tr_fot) + "/"
BEST_VERIF_RESULT_ROOT_PATH = "output/" + case_study + "/fold_" + str(baseline_fold_idx) + "_tr_" + str(num_tr_fot) + "/"
VERIF_RESULT_END_PATH = "_verif_report.csv"
mean_comfort = False
if mean_comfort:
    mean_comfort_str = " mean"
else:
    mean_comfort_str = ""

# config_str_list = ['config_0', 'config_1', 'config_2', 'config_3', 'config_4']
config_str_list = ['config_0', 'config_2', 'config_4']
plot_config_idx = [0, 2, 4]
model_str_list = ['Random', 'PR', 'RF', 'BCxGAIL_nondet_ppo']

num_y_figs = 7
num_x_figs = 3
num_max_points = 10

default_font_size = 10
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), dpi=400)

fig_idx = 0
y_fig_idx = 0

num_vis_points = 50

for model in model_str_list:
    ax_temp = axes[fig_idx]
    if not model == 'FOT':
        file_name = VERIF_RESULT_ROOT_PATH + model + VERIF_RESULT_END_PATH
        raw_df = pd.read_csv(file_name)
    else:
        file_name = VERIF_RESULT_ROOT_PATH + 'Random' + VERIF_RESULT_END_PATH
        raw_df = pd.read_csv(file_name)

    if model == 'FOT':
        baseline_file_name = BEST_VERIF_RESULT_ROOT_PATH + 'Random' + VERIF_RESULT_END_PATH
        baseline_raw_df = pd.read_csv(baseline_file_name)

        configs = raw_df['controller config'].to_numpy()
        config_idxs = get_config_idxs(configs)
        fot_safety = raw_df['fot safety'].to_numpy()
        fot_comfort = raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()

        # baseline_configs = baseline_raw_df['controller config'].to_numpy()
        # baseline_config_idxs = get_config_idxs(baseline_configs)
        best_loss_safety = baseline_raw_df['fot' + ' safety'].to_numpy()
        best_loss_comfort = baseline_raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()


    elif model == 'PR' or model == 'RF' or model == 'Random':
        configs = raw_df['controller config'].to_numpy()
        config_idxs = get_config_idxs(configs)
        fot_safety = raw_df['fot safety'].to_numpy()
        fot_comfort = raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()
        best_loss_safety = raw_df[model + ' safety'].to_numpy()
        best_loss_comfort = raw_df[model + mean_comfort_str + ' comfort'].to_numpy()
    else:
        configs = raw_df['controller config'].to_numpy()
        config_idxs = get_config_idxs(configs)
        fot_safety = raw_df['fot safety'].to_numpy()
        fot_comfort = raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()
        best_loss_safety = raw_df[model + ' best_loss' + ' safety'].to_numpy()
        best_loss_comfort = raw_df[model + ' best_loss' + mean_comfort_str + ' comfort'].to_numpy()

    if case_study == 'AV':
        fot_safety = fot_safety * 1000
        fot_comfort = (fot_comfort * 1000) / 125000
        best_loss_safety = best_loss_safety * 1000
        best_loss_comfort = (best_loss_comfort * 1000) / 125000
    else:
        fot_safety = (fot_safety * 40) / 3
        fot_comfort = (fot_comfort * 40) / (3 * 125000)
        best_loss_safety = (best_loss_safety * 40) / 3
        best_loss_comfort = (best_loss_comfort * 40) / (3 * 125000)

    safety_min = np.min(fot_safety)
    safety_max = np.max(fot_safety)
    comfort_min = np.min(fot_comfort)
    comfort_max = np.max(fot_comfort)

    configs = reshape_data_for_each_config(configs, config_idxs)
    fot_safety = reshape_data_for_each_config(fot_safety, config_idxs)
    fot_comfort = reshape_data_for_each_config(fot_comfort, config_idxs)
    best_loss_safety = reshape_data_for_each_config(best_loss_safety, config_idxs)
    best_loss_comfort = reshape_data_for_each_config(best_loss_comfort, config_idxs)

    scene_num = 1
    for c_idx in plot_config_idx:
        fot_point = ax_temp.scatter(fot_safety[c_idx, :num_vis_points], fot_comfort[c_idx, :num_vis_points], marker='x', label="Sys. ver. " + str(scene_num) + " FOT")
        color = fot_point.get_edgecolor()

        if model == 'FOT':
            ax_temp.scatter(best_loss_safety[c_idx, :num_vis_points], best_loss_comfort[c_idx, :num_vis_points], linewidths=2, marker='s', edgecolor=color, facecolor='none', label="Sys. ver. " + str(scene_num) + " new FOT")
        else:
            if model == 'PR' or model == 'RF' or model == 'Random':
                line_width = 2
            else:
                line_width = 1
            ax_temp.scatter(best_loss_safety[c_idx, :num_vis_points], best_loss_comfort[c_idx, :num_vis_points], linewidths=line_width, edgecolor=color, facecolor='none', label="Sys. ver. " + str(scene_num) + " Simulation")
        scene_num = scene_num + 1

    if model == 'BCxGAIL_nondet_ppo':
        model = 'ENVI'
    ax_temp.set_title(model, fontsize=default_font_size * 1.1)


    if case_study == 'AV':
        if not (model == 'Random'):
            None
            # ax_temp.set_xlim(safety_min - 2, safety_max + 10)
            ax_temp.set_ylim(0, 0.006) #comfort_max)
        else:
            None
    else:
        # None
        if not model == 'Random':
            ax_temp.set_xlim(safety_min - 2, safety_max + 2)
            ax_temp.set_ylim(0, comfort_max + 0.00002)
        else:
            None

    if case_study == 'Lane-keeping':
        ax_temp.set_ylabel('Maximum jerk (' + r'$mm/ms^3$' + ')', fontsize=default_font_size * 1.1)
        ax_temp.set_xlabel('Maximum displacement (' + r'$mm$' + ')', fontsize=default_font_size * 1.1)
    else:
        ax_temp.set_ylabel('Maximum jerk (' + r'$mm/ms^3$' + ')', fontsize=default_font_size * 1.1)
        ax_temp.set_xlabel('Minimum displacement (' + r'$mm$' + ')', fontsize=default_font_size * 1.1)

    fig_idx = fig_idx + 1

lines, labels = fig.axes[0].get_legend_handles_labels()
# lines1, labels1 = fig.axes[0].get_legend_handles_labels()
# lines2, labels2 = fig.axes[-1].get_legend_handles_labels()
# lines = []
# labels = []
# idx1 = 0
# idx2 = 1
# for i in range(3):
#     lines.append(lines1[idx1])
#     labels.append(labels1[idx1])
#     idx1 = idx1 + 1
#     lines.append(lines1[idx1])
#     labels.append(labels1[idx1])
#     idx1 = idx1 + 1
#     lines.append(lines2[idx2])
#     labels.append(labels2[idx2])
#     idx2 = idx2 + 2


leg = fig.legend(lines, labels, bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=3, fontsize= default_font_size * 1.0, title_fontsize=default_font_size * 1.3, frameon=True)

plt.subplots_adjust(left=0.07,
                    bottom=0.33,
                    right=0.99,
                    top=0.93,
                    wspace=0.45,
                    hspace=0.35)

# fig.suptitle('FOT and simulation trace of the environment models', fontsize = default_font_size * 1.8)
# plt.show()
plt.savefig(VERIF_RESULT_ROOT_PATH + 'scatter_comparison.png')