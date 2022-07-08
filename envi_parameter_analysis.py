import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import shapiro
from scipy.stats import anderson
from scipy.stats import levene
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu


def collect_same_tags(data_list, tag_list, tag):
    idxs = np.where(np.array(tag_list) == tag)
    same_tag_data = np.array(data_list)[idxs]
    same_tag_data = np.reshape(same_tag_data, (same_tag_data.shape[0]*same_tag_data.shape[1],))
    return same_tag_data

def mean_sort(list_of_nparr, name_list):
    stacked_nparr = np.stack(list_of_nparr)
    log_stacked_nparr = np.log(stacked_nparr)
    means = np.median(log_stacked_nparr, axis=1)
    idx_sort = np.argsort(means)
    sorted_np_arr = stacked_nparr[idx_sort]
    sorted_np_arr = [sorted_np_arr[i] for i in range(len(sorted_np_arr))]
    sorted_name_list = [name_list[i] for i in idx_sort]
    return sorted_np_arr, sorted_name_list

default_font_size = 10
label_font_rate = 1.1


# case_study = 'Lane-keeping'
case_study = 'AV'
fold_list = [0, 1, 2, 3, 4]
num_fot = 20
# columns = ['config1_kld', 'config2_kld', 'config3_kld', 'config4_kld', 'config5_kld']
columns = ['config1_kld', 'config3_kld', 'config5_kld']
# columns = ['config1_kld']
# columns = ['config3_kld']
# columns = ['config5_kld']
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


model_names = ['BC_det_loss', 'BC_det_eucl', 'BC_det_dtw',
               'BC_nondet_loss', 'BC_nondet_eucl', 'BC_nondet_dtw',
               'GAIL_det_loss', 'GAIL_det_eucl', 'GAIL_det_dtw',
               'GAIL_nondet_loss', 'GAIL_nondet_eucl', 'GAIL_nondet_dtw',
               'BCxGAIL_det_loss', 'BCxGAIL_det_eucl', 'BCxGAIL_det_dtw',
               'BCxGAIL_nondet_loss', 'BCxGAIL_nondet_eucl', 'BCxGAIL_nondet_dtw'
               ]

# model_names = model_names[3:]
kld_list = kld_list[3:]

# kld sort
kld_list, model_names = mean_sort(kld_list, model_names)
# rank_kld_list = rankdata(np.stack(kld_list), axis=0)
# kld_list = [rank_kld_list[i] for i in range(len(rank_kld_list))]
# normality = [shapiro(np.log(klds))[1] for klds in kld_list]
# normality = [anderson(np.log(klds))[1] for klds in kld_list]
# print(normality)
# print(np.mean(normality))
# print(levene(np.log(kld_list[0]), np.log(kld_list[1]), np.log(kld_list[2]), np.log(kld_list[3]), np.log(kld_list[4]), np.log(kld_list[5])))


algo_tags = []
for model_name in model_names:
    if 'BCxGAIL_' in model_name:
        algo_tags.append('BCxGAIL')
    elif 'BC_' in model_name:
        algo_tags.append('BC')
    elif 'GAIL_' in model_name:
        algo_tags.append('GAIL')
    else:
        algo_tags.append('none')

model_tags = []
for model_name in model_names:
    if '_det' in model_name:
        model_tags.append('det')
    elif '_nondet_' in model_name:
        model_tags.append('nondet')
    else:
        model_tags.append('none')

loss_tags = []
for model_name in model_names:
    if 'loss' in model_name:
        loss_tags.append('loss')
    elif 'eucl' in model_name:
        loss_tags.append('eucl')
    elif 'dtw' in model_name:
        loss_tags.append('dtw')
    else:
        loss_tags.append('none')

algo_kld_list = []
algo_list = ['BC', 'GAIL', 'BCxGAIL']
for algo_name in algo_list:
    algo_kld_list.append(collect_same_tags(kld_list, algo_tags, algo_name))

model_kld_list = []
model_list = ['det', 'nondet']
for model_name in model_list:
    model_kld_list.append(collect_same_tags(kld_list, model_tags, model_name))

loss_kld_list = []
loss_list = ['loss', 'eucl', 'dtw']
for loss_name in loss_list:
    loss_kld_list.append(collect_same_tags(kld_list, loss_tags, loss_name))



plt.style.use('seaborn-whitegrid')
fig, ax = plt.subplots(figsize=(10, 5), dpi=400)
ax.boxplot(kld_list)
ax.set_xticks(list(range(1, len(model_names)+1)), model_names, rotation=45, ha='right', rotation_mode='anchor')
ax.set_xlabel('ENVI configurations', fontsize=default_font_size * label_font_rate)
ax.set_ylabel('Imitation score (KL-divergence)', fontsize=default_font_size * label_font_rate)
ax.set_yscale('log')

plt.subplots_adjust(left=0.1,
                    bottom=0.4,
                    right=0.96,
                    top=0.93,
                    wspace=0.25,
                    hspace=0.35)

# plt.show()
plt.savefig(parent_path + str(num_fot) + "_fot_kld_boxplot.jpg")


#print(kruskal(np.log(kld_list[0]), np.log(kld_list[1]), np.log(kld_list[2]), np.log(kld_list[3]), np.log(kld_list[4]), np.log(kld_list[5])))
print('top 6 Kruskal test (p value of \"top n models have significant difference.\")')
print('top 2 p:', kruskal(kld_list[0], kld_list[1])[1])
print('top 3 p:', kruskal(kld_list[0], kld_list[1], kld_list[2])[1])
print('top 4 p:', kruskal(kld_list[0], kld_list[1], kld_list[2], kld_list[3])[1])
print('top 5 p:', kruskal(kld_list[0], kld_list[1], kld_list[2], kld_list[3], kld_list[4])[1])
print('top 6 p:', kruskal(kld_list[0], kld_list[1], kld_list[2], kld_list[3], kld_list[4], kld_list[5])[1])

print()
print('top 6 Mann Whiteney U test (p value of \"model i is better than model j.\")')
print('i|j\t1\t\t2\t\t3\t\t4\t\t5\t\t6')
for i in range(6):
    print(i+1, '\t', end='')
    for j in range(6):
        print(format(mannwhitneyu(kld_list[i], kld_list[j], alternative='less')[1], ".4f"), end='\t')
    print()

print()





figsize = (7, 5)
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=400)

width = 0.5
axes[0].boxplot(model_kld_list, widths=width)
axes[0].set_xticks(list(range(1, len(model_list)+1)), ['Deterministic', 'Nondeterministic'], rotation=45, ha='right', rotation_mode='anchor')
axes[0].set_xlabel('a) Model determinism', fontsize=default_font_size * label_font_rate)
axes[0].set_ylabel('Imitation score (KL-divergence)', fontsize=default_font_size * label_font_rate)
axes[0].set_yscale('log')

axes[1].boxplot(algo_kld_list, widths=width)
axes[1].set_xticks(list(range(1, len(algo_list)+1)), algo_list, rotation=45, ha='right', rotation_mode='anchor')
axes[1].set_xlabel('b) IL algorithm', fontsize=default_font_size * label_font_rate)
axes[1].set_ylabel('Imitation score (KL-divergence)', fontsize=default_font_size * label_font_rate)
axes[1].set_yscale('log')

axes[2].boxplot(loss_kld_list, widths=width)
axes[2].set_xticks(list(range(1, len(loss_list)+1)), ['Loss', 'Euclidean', 'DTW'], rotation=45, ha='right', rotation_mode='anchor')
axes[2].set_xlabel('c) Validation criteria', fontsize=default_font_size * label_font_rate)
axes[2].set_ylabel('Imitation score (KL-divergence)', fontsize=default_font_size * label_font_rate)
axes[2].set_yscale('log')

plt.subplots_adjust(left=0.1,
                    bottom=0.25,
                    right=0.96,
                    top=0.98,
                    wspace=0.4,
                    hspace=0.35)

fig.align_xlabels(axes)
# plt.show()
plt.savefig(parent_path + str(num_fot) + "_fot_kld_anova_boxplot.jpg")

array_for_anova_df = []
for i in range(len(kld_list)):
    for kld in kld_list[i]:
        data_item = []
        data_item.append(np.log(kld))
        # data_item.append(kld)

        if model_tags[i] == 'det':
            data_item.append(0)
        elif model_tags[i] == 'nondet':
            data_item.append(1)
        else:
            print("ERROR")
            exit()

        if algo_tags[i] == 'BC':
            data_item.append(0)
        elif algo_tags[i] == 'GAIL':
            data_item.append(1)
        elif algo_tags[i] == 'BCxGAIL':
            data_item.append(2)
        else:
            print("ERROR")
            exit()

        if loss_tags[i] == 'loss':
            data_item.append(0)
        elif loss_tags[i] == 'eucl':
            data_item.append(1)
        elif loss_tags[i] == 'dtw':
            data_item.append(2)
        else:
            print("ERROR")
            exit()

        array_for_anova_df.append(data_item)

anova_df = pd.DataFrame(array_for_anova_df, columns=['kld', 'model', 'algo', 'loss'])

import statsmodels.api as sm
from statsmodels.formula.api import ols

anova_formula = 'kld ~ C(model) * C(algo) * C(loss)'
anova_model = ols(anova_formula, anova_df).fit()
print("ANOVA (Actually, we can't perform ANOVA test here, because the data is not normally distributed.)")
print(sm.stats.anova_lm(anova_model, typ=2))
print()

print('Kruskal Test on each ENVI parameters')

model_a = anova_df.loc[anova_df['model'] == 0]['kld'].values
model_b = anova_df.loc[anova_df['model'] == 1]['kld'].values
print('model effect p value:', kruskal(model_a, model_b)[1])

algo_a = anova_df.loc[anova_df['algo'] == 0]['kld'].values
algo_b = anova_df.loc[anova_df['algo'] == 1]['kld'].values
algo_c = anova_df.loc[anova_df['algo'] == 2]['kld'].values
print('algo effect p value:', kruskal(algo_a, algo_b, algo_c)[1])

loss_a = anova_df.loc[anova_df['loss'] == 0]['kld'].values
loss_b = anova_df.loc[anova_df['loss'] == 1]['kld'].values
loss_c = anova_df.loc[anova_df['loss'] == 2]['kld'].values
print('loss effect p value:', kruskal(loss_a, loss_b, loss_c)[1])