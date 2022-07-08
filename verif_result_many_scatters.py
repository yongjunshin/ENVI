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

fold_idx = 2
num_tr_fot = 20

VERIF_RESULT_ROOT_PATH = "output/" + case_study + "/fold_" + str(fold_idx) + "_tr_" + str(num_tr_fot) + "/"
VERIF_RESULT_END_PATH = "_verif_report.csv"
mean_comfort = False
if mean_comfort:
    mean_comfort_str = " mean"
else:
    mean_comfort_str = ""

# config_str_list = ['config_0', 'config_1', 'config_2', 'config_3', 'config_4']
config_str_list = ['config_0', 'config_2', 'config_4']
model_str_list = ['Random','PR','RF',
                  'BC_det',
                  'BC_nondet',
                  'GAIL_det_ppo',
                  'GAIL_nondet_ppo',
                  'BCxGAIL_det_ppo',
                  'BCxGAIL_nondet_ppo']

num_y_figs = 7
num_x_figs = 3
num_max_points = 10

default_font_size = 10
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(num_y_figs, num_x_figs, figsize=(9, 22))

x_fig_idx = 0
y_fig_idx = 0

for model in model_str_list:
    if model == 'Random' or model == 'PR' or model == 'RF':
        ax_temp = axes[y_fig_idx, x_fig_idx]
        file_name = VERIF_RESULT_ROOT_PATH + model + VERIF_RESULT_END_PATH
        raw_df = pd.read_csv(file_name)

        configs = raw_df['controller config'].to_numpy()
        config_idxs = get_config_idxs(configs)
        fot_safety = raw_df['fot safety'].to_numpy()
        fot_comfort = raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()
        best_loss_safety = raw_df[model + ' safety'].to_numpy()
        best_loss_comfort = raw_df[model + mean_comfort_str + ' comfort'].to_numpy()

        configs = reshape_data_for_each_config(configs, config_idxs)
        fot_safety = reshape_data_for_each_config(fot_safety, config_idxs)
        fot_comfort = reshape_data_for_each_config(fot_comfort, config_idxs)
        best_loss_safety = reshape_data_for_each_config(best_loss_safety, config_idxs)
        best_loss_comfort = reshape_data_for_each_config(best_loss_comfort, config_idxs)

        colors = ['red', 'blue', 'orange', 'green', 'hotpink']
        shapes = [".", "^", "s", "x"]
        for c_idx in range(len(configs)):
            point = ax_temp.scatter(fot_safety[c_idx], fot_comfort[c_idx], marker='x',
                        label="FOT ver" + str(c_idx + 1))
            color = point.get_edgecolor()
            ax_temp.scatter(best_loss_safety[c_idx], best_loss_comfort[c_idx], edgecolor=color, facecolor='none',label="Simul ver" + str(c_idx + 1))

        ax_temp.set_title(model, fontsize=default_font_size * 1.3)
        ax_temp.set_ylabel('Comfort', fontsize=default_font_size)
        ax_temp.set_xlabel('Safety', fontsize=default_font_size)
        x_fig_idx = x_fig_idx + 1
        if x_fig_idx > num_x_figs - 1:
            x_fig_idx = 0
            y_fig_idx = y_fig_idx + 1

    else:
        file_name = VERIF_RESULT_ROOT_PATH + model + VERIF_RESULT_END_PATH
        raw_df = pd.read_csv(file_name)

        for loss_str in ['loss', 'eucl', 'dtw']:
            ax_temp = axes[y_fig_idx, x_fig_idx]

            configs = raw_df['controller config'].to_numpy()
            config_idxs = get_config_idxs(configs)
            fot_safety = raw_df['fot safety'].to_numpy()
            fot_comfort = raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()
            best_loss_safety = raw_df[model + ' best_' + loss_str + ' safety'].to_numpy()
            best_loss_comfort = raw_df[model + ' best_' + loss_str + mean_comfort_str + ' comfort'].to_numpy()

            configs = reshape_data_for_each_config(configs, config_idxs)
            fot_safety = reshape_data_for_each_config(fot_safety, config_idxs)
            fot_comfort = reshape_data_for_each_config(fot_comfort, config_idxs)
            best_loss_safety = reshape_data_for_each_config(best_loss_safety, config_idxs)
            best_loss_comfort = reshape_data_for_each_config(best_loss_comfort, config_idxs)

            for c_idx in range(len(configs)):
                point = ax_temp.scatter(fot_safety[c_idx], fot_comfort[c_idx], marker='x',
                                        label="FOT ver" + str(c_idx + 1))
                color = point.get_edgecolor()
                ax_temp.scatter(best_loss_safety[c_idx][:num_max_points], best_loss_comfort[c_idx][:num_max_points], edgecolor=color, facecolor='none',
                                label="Simul ver" + str(c_idx + 1))

            ax_temp.set_title(model+' '+loss_str, fontsize=default_font_size * 1.3)
            ax_temp.set_ylabel('Comfort', fontsize=default_font_size)
            ax_temp.set_xlabel('Safety', fontsize=default_font_size)
            x_fig_idx = x_fig_idx + 1
            if x_fig_idx > num_x_figs - 1:
                x_fig_idx = 0
                y_fig_idx = y_fig_idx + 1


fig.subplots_adjust(bottom=0.2)
lines, labels = fig.axes[-1].get_legend_handles_labels()
leg = fig.legend(lines, labels, bbox_to_anchor=(0.5, 0.05), loc='lower center', ncol=5, fontsize= default_font_size * 1.2, title_fontsize=default_font_size * 1.3, frameon=True)

plt.subplots_adjust(left=0.05,
                    bottom=0.1,
                    right=0.95,
                    top=0.93,
                    wspace=0.25,
                    hspace=0.35)

# fig.suptitle('FOT and simulation trace of the environment models', fontsize = default_font_size * 1.8)
plt.show()
# plt.savefig(VERIF_RESULT_ROOT_PATH + 'verif_vis.png')