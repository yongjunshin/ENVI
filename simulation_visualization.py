import matplotlib.pyplot as plt
import pandas as pd

case_study = 'Lane-keeping'
# case_study = 'AV'

if case_study == 'AV':
    fold_idx = 1#4
    config_str = 'config_2'#'config_4'
else:
    fold_idx = 4
    config_str = 'config_4'

num_tr_fot = 20

path = "output/" + case_study + "/fold_" + str(fold_idx) + "_tr_" + str(num_tr_fot) + "/"
trace_file_name = "simul_trace.csv"
raw_df = pd.read_csv(path + trace_file_name)


model_str_list = ['Random_loss', 'PR_loss', 'RF_loss', 'BCxGAIL_nondet_ppo_loss']
model_title_list = ['Random', 'PR', 'RF', 'ENVI']
# [FOT,Random_loss,RF_loss,PR_loss,
# BC_det_loss,BC_det_eucl,BC_det_dtw,
# BC_nondet_loss,BC_nondet_eucl,BC_nondet_dtw,
# GAIL_det_ppo_loss,GAIL_det_ppo_eucl,GAIL_det_ppo_dtw,
# GAIL_nondet_ppo_loss,GAIL_nondet_ppo_eucl,GAIL_nondet_ppo_dtw,
# BCxGAIL_det_ppo_loss,BCxGAIL_det_ppo_eucl,BCxGAIL_det_ppo_dtw,
# BCxGAIL_nondet_ppo_loss,BCxGAIL_nondet_ppo_eucl,BCxGAIL_nondet_ppo_dtw]


num_x_figs = len(model_str_list)

default_font_size = 10
plt.style.use('seaborn-whitegrid')
fig, axes = plt.subplots(1, num_x_figs, figsize=(14, 4), dpi=200)
#top_ax = fig.add_subplot(111)



FOT_trace = raw_df['FOT_' + config_str].to_numpy()
if case_study == 'Lane-keeping':
    FOT_trace = (FOT_trace * 40) / 3
else:
    FOT_trace = FOT_trace * 1000 + 1200

for x_fig_idx in range(num_x_figs):
    ax_temp = axes[x_fig_idx]
    title = model_str_list[x_fig_idx] + '_' + config_str
    simul_trace = raw_df[title].to_numpy()

    if case_study == 'Lane-keeping':
        simul_trace = (simul_trace * 40) / 3
    else:
        simul_trace = simul_trace * 1000 + 1200


    ax_temp.plot(FOT_trace, label='FOT')
    ax_temp.plot(simul_trace, label='Simulation')
    if case_study == 'AV':
        # ax_temp.set_ylim(-1, -0.4)
        ax_temp.set_ylabel('Displacement (' + r'$mm$' + ')', fontsize=default_font_size * 1.2)
        ax_temp.set_xticks([i for i in range(0, 200, 40)], [i * 50 for i in range(0, 200, 40)])
    else:
        ax_temp.set_ylabel('Displacement (' + r'$mm$' + ')', fontsize=default_font_size * 1.2)
        ax_temp.set_xticks([i for i in range(0, 100, 20)], [i * 50 for i in range(0, 100, 20)])


    ax_temp.set_title(model_title_list[x_fig_idx], fontsize = default_font_size * 1.2)


    ax_temp.set_xlabel('Time (' + r'$ms$' + ')', fontsize=default_font_size * 1.2)




fig.subplots_adjust(bottom=0.2)
lines, labels = fig.axes[-1].get_legend_handles_labels()
fig.legend(lines, labels, bbox_to_anchor=(0.5, 0.01), loc='lower center', ncol=2, frameon=True)

plt.subplots_adjust(left=0.08,
                    bottom=0.2,
                    right=0.97,
                    top=0.92,
                    wspace=0.35,
                    hspace=0.2)

# fig.supxlabel('Time (tick)', fontsize = default_font_size * 1.5)
# fig.supylabel('Lane color observed by the lane-keeping system version $\mathit{x}$', fontsize = default_font_size * 1.5)
# fig.suptitle('FOT and simulation trace of the environment models', fontsize = default_font_size * 1.8)
# plt.show()
plt.savefig(path + 'trace_vis.png')

