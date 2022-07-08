import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# case_study = 'Lane-keeping'
case_study = 'AV'
fold_list = [0, 1, 2, 3, 4]
num_fot = 20
# columns = ['config1_kld', 'config2_kld', 'config3_kld', 'config4_kld', 'config5_kld']
columns = ['config1_kld', 'config3_kld', 'config5_kld']
# columns = ['config2_kld', 'config4_kld']

parent_path = path = "output/" + case_study + "/"
kld_dataset = []
for fold_idx in fold_list:
    fold_path = "fold_" + str(fold_idx) + "_tr_" + str(num_fot) + "/"
    summary_file_name = "verif_summary.csv"
    raw_df = pd.read_csv(parent_path + fold_path + summary_file_name, index_col=False)

    model_names = raw_df['Unnamed: 0'].to_numpy()
    kld_data = raw_df[columns].to_numpy()
    kld_dataset.append(kld_data)

kld_dataset = np.concatenate(kld_dataset, axis=1)
kld_list = [list(kld_data) for kld_data in kld_dataset]


default_font_size = 10
label_font_rate = 1.1


best_kld_file_name = "output/" + case_study + "/" + str(num_fot) + "_fot_best_klds.csv"
best_klds = pd.read_csv(best_kld_file_name, header=None, index_col=False)
best_klds.dropna(axis=1, how='all', inplace=True)
best_klds = best_klds.to_numpy()
best_klds = np.reshape(best_klds, (best_klds.shape[0] * best_klds.shape[1]))




plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(3, 5), dpi=400)


model_names = [model_names[0], model_names[1], model_names[2], model_names[18]]
print(model_names)
ax.boxplot([kld_list[0], kld_list[1], kld_list[2], kld_list[18]])
ax.set_xticks(list(range(1, len(model_names)+1)), ['Random', 'PR', 'RF', 'ENVI'], rotation=45, ha='right', rotation_mode='anchor')
ax.set_xlabel('Model generation method', fontsize=default_font_size * label_font_rate)
ax.set_ylabel('Imitation score (KL-divergence)', fontsize=default_font_size * label_font_rate)
ax.set_yscale('log')

plt.subplots_adjust(left=0.2,
                    bottom=0.15,
                    right=0.95,
                    top=0.93,
                    wspace=0.25,
                    hspace=0.35)

# plt.show()
plt.savefig(parent_path + str(num_fot) + "_fot_baseline_kld_boxplot.jpg")