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


def visualize_two_dist_diff(original_dist1, dist1, original_dist2, dist2):
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
    plt.scatter(original_dist1[0], original_dist1[1], s=60, facecolors='red', edgecolors='red',
                label='fot real')

    dist2_feature1_simul_samples = []
    dist2_feature2_simul_samples = []
    for i in range(200):
        dist2_sample1 = dist2.sample()
        dist2_feature1_simul_samples.append(dist2_sample1[0].item())
        dist2_feature2_simul_samples.append(dist2_sample1[1].item())
    dist2_feature1_simul_samples = np.array(dist2_feature1_simul_samples)
    dist2_feature2_simul_samples = np.array(dist2_feature2_simul_samples)
    plt.scatter(dist2_feature1_simul_samples, dist2_feature2_simul_samples, s=60, facecolors='none', edgecolors='blue', label='simul sample')
    plt.scatter(original_dist2[0], original_dist2[1], s=60, facecolors='blue', edgecolors='blue',
                label='fot real')

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
mean_comfort = False
if mean_comfort:
    mean_comfort_str = " mean"
else:
    mean_comfort_str = ""
# fold_idx = 1
# num_tr_fot = 20
#
for fold_idx in [2]:
    for num_tr_fot in range(20, 21):
        VERIF_RESULT_ROOT_PATH = "output/" + case_study + "/fold_" + str(fold_idx) + "_tr_" + str(num_tr_fot) + "/"
        VERIF_RESULT_END_PATH = "_verif_report.csv"
        summary_file = open(VERIF_RESULT_ROOT_PATH + 'verif_summary.csv', 'w')

        algorithms = ['Random', 'PR', 'RF', 'BC_det', 'BC_nondet', 'GAIL_det_ppo', 'GAIL_nondet_ppo', 'BCxGAIL_det_ppo', 'BCxGAIL_nondet_ppo']

        mean_safety_diffs = []
        mean_comfort_diffs = []
        kl_safety_diffs = []
        kl_comfort_diffs = []

        summary_file.write(",config1_kld,config2_kld,config3_kld,config4_kld,config5_kld\n")
        for algo in algorithms:

            if algo == 'Random' or algo == 'PR' or algo == 'RF':
                print(algo)
                summary_file.write(algo + ",")
                file_name = VERIF_RESULT_ROOT_PATH + algo + VERIF_RESULT_END_PATH
                raw_df = pd.read_csv(file_name)

                configs = raw_df['controller config'].to_numpy()
                config_idxs = get_config_idxs(configs)
                fot_safety = raw_df['fot safety'].to_numpy()
                fot_comfort = raw_df['fot'+ mean_comfort_str + ' comfort'].to_numpy()
                best_loss_safety = raw_df[algo+' safety'].to_numpy()
                best_loss_comfort = raw_df[algo + mean_comfort_str + ' comfort'].to_numpy()

                configs = reshape_data_for_each_config(configs, config_idxs)
                fot_safety = reshape_data_for_each_config(fot_safety, config_idxs)
                fot_comfort = reshape_data_for_each_config(fot_comfort, config_idxs)
                best_loss_safety = reshape_data_for_each_config(best_loss_safety, config_idxs)
                best_loss_comfort = reshape_data_for_each_config(best_loss_comfort, config_idxs)

                vis_c_idx = 2
                fot_dist = get_multivariate_dist(fot_safety[vis_c_idx], fot_comfort[vis_c_idx])
                simul_dist = get_multivariate_dist(best_loss_safety[vis_c_idx], best_loss_comfort[vis_c_idx])
                visualize_two_dist_diff((fot_safety[vis_c_idx], fot_comfort[vis_c_idx]), fot_dist, (best_loss_safety[vis_c_idx], best_loss_comfort[vis_c_idx]), simul_dist)

                colors = ['red', 'blue', 'orange', 'green', 'hotpink']
                shapes = [".", "^", "s", "x"]
                plt.figure(figsize=(9, 6))
                for c_idx in range(len(configs)):
                    plt.scatter(fot_safety[c_idx], fot_comfort[c_idx], s=60, facecolors=colors[c_idx], edgecolors=colors[c_idx],
                                label="FOT x=" + str((c_idx + 1) * 10))
                    plt.scatter(best_loss_safety[c_idx], best_loss_comfort[c_idx], marker="s", s=40, facecolors='none', edgecolors=colors[c_idx], label="best_loss_model")
                    # plt.scatter(best_eucl_safety[c_idx], best_eucl_comfort[c_idx], marker="D", s=40, facecolors='none',
                    #             edgecolors=colors[c_idx], label="best_eucl_model")
                    # plt.scatter(best_dtw_safety[c_idx], best_dtw_comfort[c_idx], marker="^", s=40, facecolors='none',
                    #             edgecolors=colors[c_idx], label="best_dtw_model")

                    fot_dist = get_multivariate_dist(fot_safety[c_idx], fot_comfort[c_idx])
                    simul_dist = get_multivariate_dist(best_loss_safety[c_idx], best_loss_comfort[c_idx])
                    print("config", c_idx+1, "th KL divergence:", kl_divergence(fot_dist, simul_dist))
                    summary_file.write(str(kl_divergence(fot_dist, simul_dist).item()) + ",")
                summary_file.write("\n")



                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.xlabel("safety")
                plt.ylabel("passenger comfort")
                plt.title(algo + " model simulation results")
                plt.show()

            else:
                print(algo)
                file_name = VERIF_RESULT_ROOT_PATH + algo + VERIF_RESULT_END_PATH
                raw_df = pd.read_csv(file_name)

                configs = raw_df['controller config'].to_numpy()
                config_idxs = get_config_idxs(configs)
                fot_safety = raw_df['fot safety'].to_numpy()
                fot_comfort = raw_df['fot' + mean_comfort_str + ' comfort'].to_numpy()
                best_loss_safety = raw_df[algo + ' best_loss safety'].to_numpy()
                best_loss_comfort = raw_df[algo + ' best_loss' + mean_comfort_str + ' comfort'].to_numpy()
                best_eucl_safety = raw_df[algo + ' best_eucl safety'].to_numpy()
                best_eucl_comfort = raw_df[algo + ' best_eucl' + mean_comfort_str + ' comfort'].to_numpy()
                best_dtw_safety = raw_df[algo + ' best_dtw safety'].to_numpy()
                best_dtw_comfort = raw_df[algo + ' best_dtw' + mean_comfort_str + ' comfort'].to_numpy()

                configs = reshape_data_for_each_config(configs, config_idxs)
                fot_safety = reshape_data_for_each_config(fot_safety, config_idxs)
                fot_comfort = reshape_data_for_each_config(fot_comfort, config_idxs)
                best_loss_safety = reshape_data_for_each_config(best_loss_safety, config_idxs)
                best_loss_comfort = reshape_data_for_each_config(best_loss_comfort, config_idxs)
                best_eucl_safety = reshape_data_for_each_config(best_eucl_safety, config_idxs)
                best_eucl_comfort = reshape_data_for_each_config(best_eucl_comfort, config_idxs)
                best_dtw_safety = reshape_data_for_each_config(best_dtw_safety, config_idxs)
                best_dtw_comfort = reshape_data_for_each_config(best_dtw_comfort, config_idxs)

                # vis_c_idx = 0
                # fot_dist = get_multivariate_dist(fot_safety[vis_c_idx], fot_comfort[vis_c_idx])
                # simul_dist = get_multivariate_dist(best_loss_safety[vis_c_idx], best_loss_comfort[vis_c_idx])
                # visualize_two_dist_diff(fot_dist, simul_dist)

                colors = ['red', 'blue', 'orange', 'green', 'hotpink']
                shapes = [".", "^", "s", "x"]
                plt.figure(figsize=(9, 6))
                for c_idx in range(len(configs)):
                    plt.scatter(fot_safety[c_idx], fot_comfort[c_idx], s=60, facecolors=colors[c_idx], edgecolors=colors[c_idx], label="FOT x=" + str((c_idx+1)*10))
                    plt.scatter(best_loss_safety[c_idx], best_loss_comfort[c_idx], marker="s", s=40, facecolors='none', edgecolors=colors[c_idx], label="best_loss_model")
                    plt.scatter(best_eucl_safety[c_idx], best_eucl_comfort[c_idx], marker="D", s=40, facecolors='none', edgecolors=colors[c_idx], label="best_eucl_model")
                    plt.scatter(best_dtw_safety[c_idx], best_dtw_comfort[c_idx], marker="^", s=40, facecolors='none', edgecolors=colors[c_idx], label="best_dtw_model")

                    fot_dist = get_multivariate_dist(fot_safety[c_idx], fot_comfort[c_idx])
                    best_loss_simul_dist = get_multivariate_dist(best_loss_safety[c_idx], best_loss_comfort[c_idx])
                    best_eucl_simul_dist = get_multivariate_dist(best_eucl_safety[c_idx], best_eucl_comfort[c_idx])
                    best_dtw_simul_dist = get_multivariate_dist(best_dtw_safety[c_idx], best_dtw_comfort[c_idx])

                    print("best_loss, config", c_idx + 1, "th KL divergence:", kl_divergence(fot_dist, best_loss_simul_dist))
                    print("best_eucl, config", c_idx + 1, "th KL divergence:", kl_divergence(fot_dist, best_eucl_simul_dist))
                    print("best_dtw, config", c_idx + 1, "th KL divergence:", kl_divergence(fot_dist, best_dtw_simul_dist))
                    print()

                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.xlabel("safety")
                plt.ylabel("passenger comfort")
                plt.title(algo + " model simulation results")
                plt.show()

                summary_file.write(algo + "_best_loss,")
                for c_idx in range(len(configs)):
                    fot_dist = get_multivariate_dist(fot_safety[c_idx], fot_comfort[c_idx])
                    simul_dist = get_multivariate_dist(best_loss_safety[c_idx], best_loss_comfort[c_idx])
                    summary_file.write(str(kl_divergence(fot_dist, simul_dist).item()) + ",")
                summary_file.write("\n")

                summary_file.write(algo + "_best_eucl,")
                for c_idx in range(len(configs)):
                    fot_dist = get_multivariate_dist(fot_safety[c_idx], fot_comfort[c_idx])
                    simul_dist = get_multivariate_dist(best_eucl_safety[c_idx], best_eucl_comfort[c_idx])
                    summary_file.write(str(kl_divergence(fot_dist, simul_dist).item()) + ",")
                summary_file.write("\n")

                summary_file.write(algo + "_best_dtw,")
                for c_idx in range(len(configs)):
                    fot_dist = get_multivariate_dist(fot_safety[c_idx], fot_comfort[c_idx])
                    simul_dist = get_multivariate_dist(best_dtw_safety[c_idx], best_dtw_comfort[c_idx])
                    summary_file.write(str(kl_divergence(fot_dist, simul_dist).item()) + ",")
                summary_file.write("\n")


        summary_file.close()