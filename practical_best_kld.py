import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
num_fot = 20

mean_comfort = False
if mean_comfort:
    mean_comfort_str = " mean"
else:
    mean_comfort_str = ""
# fold_idx = 1
# num_tr_fot = 20
#
summary_file_name = "output/" + case_study + "/" + str(num_fot) + "_fot_best_klds.csv"
summary_file = open(summary_file_name, 'w')
for fold_idx_i in [0, 1, 2, 3, 4]:
    for fold_idx_j in [0, 1, 2, 3, 4]:
        print(fold_idx_i, fold_idx_j)
        if fold_idx_i >= fold_idx_j:
            continue
        else:
            source_file_name = "output/" + case_study + "/fold_" + str(fold_idx_i) + "_tr_" + str(num_fot) + "/Random_verif_report.csv"
            source_raw_df = pd.read_csv(source_file_name)
            source_configs = source_raw_df['controller config'].to_numpy()
            source_config_idxs = get_config_idxs(source_configs)
            source_fot_safety = source_raw_df['fot safety'].to_numpy()
            source_fot_comfort = source_raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()
            source_configs = reshape_data_for_each_config(source_configs, source_config_idxs)
            source_fot_safety = reshape_data_for_each_config(source_fot_safety, source_config_idxs)
            source_fot_comfort = reshape_data_for_each_config(source_fot_comfort, source_config_idxs)

            target_file_name = "output/" + case_study + "/fold_" + str(fold_idx_j) + "_tr_" + str(num_fot) + "/Random_verif_report.csv"
            target_raw_df = pd.read_csv(target_file_name)
            target_configs = target_raw_df['controller config'].to_numpy()
            target_config_idxs = get_config_idxs(target_configs)
            target_fot_safety = target_raw_df['fot safety'].to_numpy()
            target_fot_comfort = target_raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()
            target_configs = reshape_data_for_each_config(target_configs, target_config_idxs)
            target_fot_safety = reshape_data_for_each_config(target_fot_safety, target_config_idxs)
            target_fot_comfort = reshape_data_for_each_config(target_fot_comfort, target_config_idxs)

            colors = ['red', 'blue', 'orange', 'green', 'hotpink']
            shapes = [".", "^", "s", "x"]
            plt.figure(figsize=(9, 6))
            klds = []
            for c_idx in [0, 2, 4]:
                source_fot_dist = get_multivariate_dist(source_fot_safety[c_idx], source_fot_comfort[c_idx])
                target_fot_dist = get_multivariate_dist(target_fot_safety[c_idx], target_fot_comfort[c_idx])
                plt.scatter(source_fot_safety[c_idx], source_fot_comfort[c_idx], s=60, facecolors=colors[c_idx], edgecolors=colors[c_idx], label="FOT x=" + str((c_idx + 1) * 10))
                plt.scatter(target_fot_safety[c_idx], target_fot_comfort[c_idx], marker="s", s=40, facecolors='none', edgecolors=colors[c_idx], label="best_loss_model")

                kld = kl_divergence(source_fot_dist, target_fot_dist)
                print("config", c_idx + 1, "th KL divergence:", kld)
                klds.append(kld)

                summary_file.write(str(kld.item()) + ",")

            plt.title(str(klds[0].item())+' '+str(klds[1].item())+' '+str(klds[2].item()))
            # plt.show()

            summary_file.write("\n")
summary_file.close()
