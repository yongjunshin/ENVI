import pandas as pd
import numpy as np

case_study = 'Lane-keeping'
# case_study = 'AV'
fold_list = [0, 1, 2, 3, 4]
num_fot = 20
columns = ['config1_kld', 'config2_kld', 'config3_kld', 'config4_kld', 'config5_kld']

parent_path = path = "output/" + case_study + "/"
dataset = []
for fold_idx in fold_list:
    fold_path = "fold_" + str(fold_idx) + "_tr_" + str(num_fot) + "/"
    summary_file_name = "verif_summary.csv"
    raw_df = pd.read_csv(parent_path + fold_path + summary_file_name, index_col=False)

    model_names = raw_df['Unnamed: 0'].to_numpy()
    log_data = np.log(raw_df[columns].to_numpy())
    dataset.append(log_data)

dataset = np.stack(dataset)
avg_data = np.mean(dataset, axis=0)
avg_data = np.exp(avg_data)

column_avg = np.mean(avg_data, axis=1)
column_avg = np.reshape(column_avg, (column_avg.shape[0], 1))

final_data = np.concatenate((avg_data, column_avg), axis=1)
None

columns.append('avg')

avg_df = pd.DataFrame(data=final_data,
                      index=model_names,
                      columns=columns)
avg_df.to_csv(parent_path + str(num_fot) + "_fot_kld_avg.csv")

