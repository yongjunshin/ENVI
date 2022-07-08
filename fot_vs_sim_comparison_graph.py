import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata


def reshape_data_for_each_config(data, config_idxs_list):
    reshaped_data = []
    for config_idxs in config_idxs_list:
        reshaped_data.append(data[config_idxs])
    reshaped_data = np.stack(reshaped_data)
    return reshaped_data


def get_config_idxs(configs_arr):
    unique_configs = []
    idxs = []
    for i in range(len(configs_arr)):
        if not configs_arr[i] in unique_configs:
            unique_configs.append(configs_arr[i])
            idxs.append(np.where(configs_arr == configs_arr[i]))
    return idxs


case_study = 'Lane-keeping'
# case_study = 'AV'
mean_comfort = False
if mean_comfort:
    mean_comfort_str = " mean"
else:
    mean_comfort_str = ""


# vis_prop = 'safety'
vis_prop = 'comfort'

num_tr_fot = 20
if case_study == 'Lane-keeping':
    fold_list = [0, 1, 2, 3, 4]
else:
    fold_list = [0, 1, 2]
model_list = ['PR', 'RF', 'BC_nondet', 'BCxGAIL_nondet_ppo']

median_flag = False
num_vis_points = 10

num_x_figs = len(model_list)

default_font_size = 10
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(1, num_x_figs, figsize=(18, 5))

fig_idx = 0
parent_path = 'output/' + case_study + '/'
for model in model_list:
    ax_temp = axes[fig_idx]
    if model == 'PR' or model == 'RF':
        loss_name = ''
    else:
        loss_name = 'best_loss '


    fot_list = []
    sim_list = []
    for fold in fold_list:
        fold_path = 'fold_' + str(fold) + '_tr_' + str(num_tr_fot) + '/'
        file_name = parent_path + fold_path + model + '_verif_report.csv'
        raw_df = pd.read_csv(file_name)

        configs = raw_df['controller config'].to_numpy()
        config_idxs = get_config_idxs(configs)

        fot_all_configs = raw_df['fot ' + vis_prop].to_numpy()
        fot_seperate_configs = reshape_data_for_each_config(fot_all_configs, config_idxs)
        if median_flag:
            fot_seperate_configs = fot_seperate_configs[:, :num_vis_points]
        fot_mean_for_each_config = np.mean(fot_seperate_configs, axis=1)



        simul_all_configs = raw_df[model + ' ' + loss_name + vis_prop].to_numpy()
        simul_seperate_configs = reshape_data_for_each_config(simul_all_configs, config_idxs)
        if median_flag:
            simul_seperate_configs = np.sort(simul_seperate_configs, axis=1)
            median_start_idx = int(simul_seperate_configs.shape[1]/2 - num_vis_points/2)
            simul_seperate_configs = simul_seperate_configs[:, median_start_idx:median_start_idx + num_vis_points]
        simul_mean_for_each_config = np.mean(simul_seperate_configs, axis=1)




        fot_list.append(fot_mean_for_each_config)
        sim_list.append(simul_mean_for_each_config)

    fot_list = np.concatenate(fot_list, axis=0)
    sim_list = np.concatenate(sim_list, axis=0)

    p_value = np.corrcoef(fot_list, sim_list)[0, 1]
    fot_rank = rankdata(fot_list)
    simul_rank = rankdata(sim_list)
    spearman_value = np.corrcoef(fot_rank, simul_rank)[0, 1]

    print(model)
    print('p:', p_value)
    print('spearman:', spearman_value)

    ax_temp.scatter(fot_list, sim_list)
    ax_temp.set_title(model)
    ax_temp.set_xlabel("FOT result")
    ax_temp.set_ylabel("Simulation result")

    fig_idx = fig_idx + 1

plt.show()