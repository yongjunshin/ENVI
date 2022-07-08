import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# case_study = 'Lane-keeping'
case_study = 'AV'


fold_idx_list = [0, 1, 2, 3, 4]


num_tr_fot_list = range(1, 21)
num_models = 21
print_num_tr_fot = [1, 10, 20]

default_font_size = 10

efficiency_data = []
random_data = []
parent_path = 'output/' + case_study + '/'
for num_tr_fot in num_tr_fot_list:
    model_data_scenario_summary_avg = np.zeros(num_models)
    for fold_idx in fold_idx_list:
        verif_summary_fname = parent_path + "fold_" + str(fold_idx) + "_tr_" + str(num_tr_fot) + "/verif_summary.csv"
        raw_df = pd.read_csv(verif_summary_fname, index_col=False)
        verif_summary_data = raw_df.to_numpy()

        model_names = verif_summary_data[:, 0]
        model_data = verif_summary_data[:, 1:].astype(np.float32)
        model_data_scenario_summary = np.median(np.log(model_data[:, [0, 2, 4]]), axis=1)
        if num_tr_fot == 20:
            random_data.append(model_data[0, 0])
            random_data.append(model_data[0, 2])
            random_data.append(model_data[0, 4])

        model_data_scenario_summary_avg = model_data_scenario_summary_avg + model_data_scenario_summary


    model_data_scenario_summary_avg = model_data_scenario_summary_avg / len(fold_idx_list)
    model_data_scenario_summary_avg = np.exp(model_data_scenario_summary_avg)

    if num_tr_fot in print_num_tr_fot:
        print("num training data: ", num_tr_fot)
        print(model_data_scenario_summary_avg)

    efficiency_data.append(model_data_scenario_summary_avg)

efficiency_data = np.stack(efficiency_data)
print(model_names)

best_kld_file_name = "output/" + case_study + "/" +"20_fot_best_klds.csv"
best_klds = pd.read_csv(best_kld_file_name, header=None, index_col=False)
best_klds.dropna(axis=1, how='all', inplace=True)
best_klds = best_klds.to_numpy()
best_klds = np.reshape(best_klds, (best_klds.shape[0] * best_klds.shape[1]))

fot_upper_quartile = np.percentile(best_klds, 75)
fot_lower_quartile = np.percentile(best_klds, 25)
iqr = fot_upper_quartile - fot_lower_quartile
fot_upper_whisker = best_klds[best_klds <= fot_upper_quartile + 1.5 * iqr].max()
# fot_criteria = fot_upper_whisker
fot_criteria = fot_upper_quartile

random_data = np.array(random_data)
random_upper_quartile = np.percentile(random_data, 75)
random_lower_quartile = np.percentile(random_data, 25)
iqr = random_upper_quartile - random_lower_quartile
random_lower_whisker = random_data[random_data >= random_lower_quartile-1.5*iqr].min()
# random_criteria = random_lower_whisker
# random_criteria = random_lower_quartile
random_criteria = np.median(random_data)

plot_lines = []
plt.figure(figsize=(3, 6), dpi=400)
plt.style.use('seaborn-whitegrid')

# plt.axhline(y=fot_criteria, linestyle='--', color='purple', label='FOT (75 percentile)')
plt.axhline(y=random_criteria, linestyle='--', color='r', label='Random')

for model_idx in range(len(model_names)):
    if model_idx in [1, 2, 18]:
        # print(model_names[model_idx])
        # print(efficiency_data[:, model_idx])
        # if model_idx < 3:
        #     marker = None
        # elif model_idx < 9:
        #     marker = "s"
        # elif model_idx < 15:
        #     marker = "^"
        # else:
        #     marker = "o"
        if model_names[[model_idx]] == 'BCxGAIL_nondet_ppo_best_loss':
            plt.plot(efficiency_data[:, model_idx], label='ENVI')
        else:
            plt.plot(efficiency_data[:, model_idx], label=model_names[model_idx])

default_x_ticks = list(range(len(num_tr_fot_list)))
new_x_ticks = list(num_tr_fot_list)
plt.xticks(default_x_ticks, new_x_ticks, fontsize=default_font_size*1.1)
plt.legend(fontsize=default_font_size*1.1)
# plt.title("Training data efficiency of environment model generation", fontsize=default_font_size*1.2)
plt.yscale("log")
plt.ylabel("Imitation score (KL-divergence)", fontsize=default_font_size*1.2)
plt.xlabel("Number of training FOT logs", fontsize=default_font_size*1.2)
plt.locator_params(axis='x', nbins=10)

plt.subplots_adjust(left=0.2,
                    bottom=0.1,
                    right=0.97,
                    top=0.92,
                    wspace=0.35,
                    hspace=0.2)


# plt.show()
plt.savefig(parent_path + "efficiency.jpg")



