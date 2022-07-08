import os
from src.av_controller import AV
from src.av_env_model import AVEnvModelDNN
from src.bc import BCTrainer
from src.data_manager import *
from src.envi_config import EnviConfig
from src.envi_util import load_best_model, draw_plot, save_best_model, save_model
from src.gail import GAILTrainer
from src.gail_actor_critic import GAILActorCriticTrainer
from src.gail_integrated import GAILIntegration
from src.random_model import RandomEnvironmentModelDNN
from src.simulation import non_deterministic_simulation, get_dtw, get_euclidean
from src.simulation import deterministic_simulation
from src.gail_reinforce import GAILReinforceTrainer
from src.data_manager_lk_fot import load_tmvk_norm_np_data, load_tmvu_norm_np_data, np_log_to_tensor_training_data
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from src.gail_tdac import GAILTdacTrainer
from src.goal_verification import *
from src.lk_controller import LaneKeepingSystem
from src.shallow_model import LineTracerShallowRFEnvironmentModel, LineTracerShallowPREnvironmentModel
from src.training_dataset import TrainingDataset

# fixed setting
FOT_LOG_DIR = "data/Lane-keeping-FOT/"
TR_FLAG = 0
VA_FLAG = 1
TE_FLAG = 2
COLUMNS = ['color', 'angle']
STATE_FEATURES = 1
ACTION_FEATURES = 1
FC_LAYER_FEATURES = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
COLOR_FEATURE_IDX = 0
DENORM_LANE_KEEPING_COLOR_GOAL = 40
NUM_FOLD = 5

for split_tr_FOTs in range(20, 21):
    # user setting
    configs = [[10], [20], [30], [40], [50]]
    tr_config_idxs = [0, 2, 4]
    va_config_idxs = [0, 2, 4]
    te_config_idxs = [0, 1, 2, 3, 4]
    split_nums = [split_tr_FOTs, split_tr_FOTs, 10]  # [20, 19, 10]
    if split_tr_FOTs == 20:
        split_nums = [split_tr_FOTs, split_tr_FOTs-1, 10]
    current_fold_idx = 0   # less than NUM_FOLD   #todo
    history_len = 10
    common_lr = 0.00005
    common_epoch = 500
    common_model_save_period = 5

    current_trial_id = 'fold_' + str(current_fold_idx) + '_tr_' + str(split_nums[0])
    new_directory = 'D:/ENVI/output/Lane-keeping/temp/' + current_trial_id

    try:
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
    except OSError:
        print('Error: Creating directory. ' + new_directory)

    exp_configs = []
    exp_configs.append(EnviConfig(envi_algo='random', lr=0, epoch=0))
    exp_configs.append(EnviConfig(envi_algo='rf', lr=0, epoch=0))
    exp_configs.append(EnviConfig(envi_algo='pr', lr=0, epoch=0))
    exp_configs.append(EnviConfig(envi_algo='bc', lr=common_lr, epoch=common_epoch, bc_deterministic=True, model_save_period=common_model_save_period))
    exp_configs.append(EnviConfig(envi_algo='bc', lr=common_lr, epoch=common_epoch, bc_deterministic=False, model_save_period=common_model_save_period))
    # exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=False, gail_algo='reinforce', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))
    # exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=False, gail_algo='actor_critic', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))
    exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=False, gail_deterministic=True, gail_algo='ppo', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))
    exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=False, gail_deterministic=False, gail_algo='ppo', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))
    # exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=True, gail_algo='reinforce', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))
    # exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=True, gail_algo='actor_critic', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))
    exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=True, gail_deterministic=True, gail_algo='ppo', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))
    exp_configs.append(EnviConfig(envi_algo='gail', lr=common_lr, epoch=common_epoch, gail_x_bc=True, gail_deterministic=False, gail_algo='ppo', gail_disc_learning_break=9, gail_disc_iter=10, gail_ppo_iter=10, model_save_period=common_model_save_period))

    # begin dataset loading
    configs_string = []
    for config in configs:
        configs_string.append(str(config[0]))

    np_data, scaler = prepare_np_fot_data(FOT_LOG_DIR, configs_string, COLUMNS, split_nums, NUM_FOLD, current_fold_idx)

    tr_np_log, tr_np_c = collect_np_data_and_config_for_purpose(np_data, configs, tr_config_idxs, TR_FLAG)
    va_np_log, va_np_c = collect_np_data_and_config_for_purpose(np_data, configs, va_config_idxs, VA_FLAG)
    te_np_log, te_np_c = collect_np_data_and_config_for_purpose(np_data, configs, te_config_idxs, TE_FLAG)

    tr_tensor_log = torch.tensor(np.stack(tr_np_log), dtype=torch.float32, device=DEVICE)
    va_tensor_log = torch.tensor(np.stack(va_np_log), dtype=torch.float32, device=DEVICE)
    te_tensor_log = torch.tensor(np.stack(te_np_log), dtype=torch.float32, device=DEVICE)

    tr_tensor_x, tr_tensor_y, tr_tensor_c, tr_tensor_x_init, tr_tensor_y_init, tr_tensor_c_init = np_to_tensor_xyc_data(tr_np_log, tr_np_c, history_len, STATE_FEATURES, DEVICE)
    va_tensor_x, va_tensor_y, va_tensor_c, va_tensor_x_init, va_tensor_y_init, va_tensor_c_init = np_to_tensor_xyc_data(va_np_log, va_np_c, history_len, STATE_FEATURES, DEVICE)
    te_tensor_x, te_tensor_y, te_tensor_c, te_tensor_x_init, te_tensor_y_init, te_tensor_c_init = np_to_tensor_xyc_data(te_np_log, te_np_c, history_len, STATE_FEATURES, DEVICE)

    tr_dataset = TrainingDataset(tr_np_log, tr_tensor_log, tr_tensor_x, tr_tensor_y, tr_tensor_c, tr_tensor_x_init, tr_tensor_y_init, tr_tensor_c_init,
                                 va_np_log, va_tensor_log, va_tensor_x, va_tensor_y, va_tensor_c, va_tensor_x_init, va_tensor_y_init, va_tensor_c_init)
    # end dataset loading

    simul_traces_name_for_vis = []
    simul_traces_for_vis = []
    simul_traces_file = open(new_directory + '/simul_trace.csv', 'w')
    for te_config_name in te_config_idxs:
        simul_traces_name_for_vis.append('FOT' + "_config_" + str(te_config_name))
    for te_vis_config_idx in range(len(te_config_idxs)):
        simul_traces_for_vis.append(te_tensor_log[te_vis_config_idx * split_nums[2],:,[0]].cpu().numpy())

    for exp_config in exp_configs:
        # begin model learning
        av = LaneKeepingSystem(scaler)
        env_model = AVEnvModelDNN(STATE_FEATURES + ACTION_FEATURES, history_len, FC_LAYER_FEATURES, STATE_FEATURES, DEVICE)
        best_models = []

        envi_algorithm = exp_config.envi_algorithm

        verif_output_path = "output/Lane-keeping/temp/" + current_trial_id + "/"
        temp_model_path = "output/Lane-keeping/temp_models/"

        if envi_algorithm == 'random':
            model_prefix = "Random"
            print(model_prefix)
            simul_deterministic_flag = True

            flat_tr_x = torch.reshape(tr_tensor_x,
                                      (tr_tensor_x.shape[0], tr_tensor_x.shape[1] * tr_tensor_x.shape[2])).cpu().numpy()
            flat_tr_y = tr_tensor_y.cpu().numpy()
            flat_tr_y = np.reshape(flat_tr_y, (flat_tr_y.shape[0]))

            poly_features = PolynomialFeatures()
            poly_flat_tr_x = poly_features.fit_transform(flat_tr_x)

            linear_regressor = LinearRegression()
            linear_regressor.fit(poly_flat_tr_x, flat_tr_y)
            env_model = RandomEnvironmentModelDNN(STATE_FEATURES, DEVICE)
            # save_best_model(env_model, best_model_dir_path + "best_random_model.pt")

            best_models = [env_model]
            best_models_names = [model_prefix]

        elif envi_algorithm == 'bc':
            # BC
            lr = exp_config.lr
            epoch = exp_config.epoch

            bc_deterministic_model_flag = exp_config.bc_determinisitic
            bc_model_save_period = exp_config.model_save_period

            if bc_deterministic_model_flag:
                model_prefix = "BC_det"
                simul_deterministic_flag = True
            else:
                model_prefix = "BC_nondet"
                simul_deterministic_flag = False
            print(model_prefix)

            bc = BCTrainer(model_prefix, DEVICE, av, lr)

            bc_model_epch_l, bc_tr_l, bc_va_l, bc_tr_eucl_l, bc_va_eucl_l, bc_tr_dtw_l, bc_va_dtw_l = bc.train(env_model, epoch, tr_dataset, temp_model_path, bc_deterministic_model_flag, bc_model_save_period)
            report = [bc_model_epch_l, bc_tr_l, bc_va_l, bc_tr_eucl_l, bc_va_eucl_l, bc_tr_dtw_l, bc_va_dtw_l]
            report_df = pd.DataFrame(np.array(report).T, columns=['epoch', 't_loss', 'v_loss', 't_eucl', 'v_eucl', 't_dtw', 'v_dtw'], dtype=float)
            report_df.to_csv(temp_model_path + model_prefix + "_training_report.csv")

            best_loss_bc_model = load_best_model(env_model, temp_model_path, model_prefix, bc_va_l, bc_model_epch_l)
            #save_best_model(best_loss_bc_model, best_model_dir_path + "best_loss_model.pt")

            best_eucl_bc_model = load_best_model(env_model, temp_model_path, model_prefix, bc_va_eucl_l, bc_model_epch_l)
            #save_best_model(best_eucl_bc_model, best_model_dir_path + "best_eucl_model.pt")

            best_dtw_bc_model = load_best_model(env_model, temp_model_path, model_prefix, bc_va_dtw_l, bc_model_epch_l)
            #save_best_model(best_dtw_bc_model, best_model_dir_path + "best_dtw_model.pt")

            best_models = [best_loss_bc_model, best_eucl_bc_model, best_dtw_bc_model]
            best_models_names = [model_prefix + ' best_loss', model_prefix + ' best_eucl', model_prefix + ' best_dtw']

            bc_model_save_epoches = range(0, epoch, bc_model_save_period)
            # draw_plot(model_prefix + " loss", [bc_tr_l, bc_va_l], ["tr_loss", "va_loss"], x_axis=bc_model_save_epoches)
            # draw_plot(model_prefix + " eucl", [bc_tr_eucl_l, bc_va_eucl_l], ["tr_eucl_loss", "va_eucl_loss"], x_axis=bc_model_save_epoches)
            # draw_plot(model_prefix + " dtw", [bc_tr_dtw_l, bc_va_dtw_l], ["tr_dtw_loss", "va_dtw_loss"], x_axis=bc_model_save_epoches)

        elif envi_algorithm == 'gail':
            # GAIL
            lr = exp_config.lr
            epoch = exp_config.epoch
            gail_deterministic = exp_config.gail_deterministic
            gail_x_bc = exp_config.gail_x_bc
            gail_algorithm = exp_config.gail_algo
            disc_learning_break = exp_config.gaIl_disc_learning_break
            disc_iter = exp_config.gaIl_disc_iter
            ppo_iter = exp_config.gail_ppo_iter
            gail_model_save_period = exp_config.model_save_period

            if gail_x_bc:
                if gail_deterministic:
                    model_prefix = 'BCxGAIL_det_' + gail_algorithm
                    simul_deterministic_flag = True
                else:
                    model_prefix = 'BCxGAIL_nondet_' + gail_algorithm
                    simul_deterministic_flag = False
            else:
                if gail_deterministic:
                    model_prefix = 'GAIL_det_' + gail_algorithm
                    simul_deterministic_flag = True
                else:
                    model_prefix = 'GAIL_nondet_' + gail_algorithm
                    simul_deterministic_flag = False
            print(model_prefix)

            gail = GAILIntegration(model_prefix, DEVICE, av, STATE_FEATURES, ACTION_FEATURES, history_len, lr, disc_learning_break, disc_iter, ppo_iter, gail_deterministic)

            gail_model_epch_l, gail_tr_l, gail_va_l, gail_tr_eucl_l, gail_va_eucl_l, gail_tr_dtw_l, gail_va_dtw_l = gail.train(env_model, epoch, tr_dataset, temp_model_path, gail_x_bc, gail_algorithm, gail_model_save_period)
            report = [gail_model_epch_l, gail_tr_l, gail_va_l, gail_tr_dtw_l, gail_va_dtw_l, gail_tr_eucl_l, gail_va_eucl_l]
            report_df = pd.DataFrame(np.array(report).T, columns=['epoch', 't_loss', 'v_loss', 't_eucl', 'v_eucl', 't_dtw', 'v_dtw'], dtype=float)
            report_df.to_csv(temp_model_path + model_prefix + "_training_report.csv")

            best_loss_gail_model = load_best_model(env_model, temp_model_path, model_prefix, gail_va_l, gail_model_epch_l)
            #save_best_model(best_loss_gail_model, best_model_dir_path + "best_loss_model.pt")

            best_eucl_gail_model = load_best_model(env_model, temp_model_path, model_prefix, gail_va_eucl_l, gail_model_epch_l)
            #save_best_model(best_eucl_gail_model, best_model_dir_path + "best_eucl_model.pt")

            best_dtw_gail_model = load_best_model(env_model, temp_model_path, model_prefix, gail_va_dtw_l, gail_model_epch_l)
            #save_best_model(best_dtw_gail_model, best_model_dir_path + "best_dtw_model.pt")

            best_models = [best_loss_gail_model, best_eucl_gail_model, best_dtw_gail_model]
            best_models_names = [model_prefix + ' best_loss', model_prefix + ' best_eucl', model_prefix + ' best_dtw']

            gail_model_save_epoches = range(0, epoch, gail_model_save_period)
            # draw_plot(model_prefix + " loss", [gail_tr_l, gail_va_l], ["tr_loss", "va_loss"], x_axis=gail_model_save_epoches)
            # draw_plot(model_prefix + " eucl", [gail_tr_eucl_l, gail_va_eucl_l], ["tr_eucl_loss", "va_eucl_loss"], x_axis=gail_model_save_epoches)
            # draw_plot(model_prefix + " dtw", [gail_tr_dtw_l, gail_va_dtw_l], ["tr_dtw_loss", "va_dtw_loss"], x_axis=gail_model_save_epoches)

        elif envi_algorithm == 'pr':
            model_prefix = 'PR'
            print(model_prefix)

            flat_tr_x = torch.reshape(tr_tensor_x,
                                      (tr_tensor_x.shape[0], tr_tensor_x.shape[1] * tr_tensor_x.shape[2])).cpu().numpy()
            flat_tr_y = tr_tensor_y.cpu().numpy()
            flat_tr_y = np.reshape(flat_tr_y, (flat_tr_y.shape[0]))

            poly_features = PolynomialFeatures()
            poly_flat_tr_x = poly_features.fit_transform(flat_tr_x)

            linear_regressor = LinearRegression()
            linear_regressor.fit(poly_flat_tr_x, flat_tr_y)
            shallow_PR_model = LineTracerShallowPREnvironmentModel(linear_regressor, poly_features, STATE_FEATURES, DEVICE)
            env_model = shallow_PR_model
            save_model(env_model, temp_model_path, model_prefix, 0)
            #save_best_model(env_model, best_model_dir_path + "best_pr_model.pt")

            best_models = [env_model]
            best_models_names = [model_prefix]

        elif envi_algorithm == 'rf':
            model_prefix = 'RF'
            print(model_prefix)

            rf_regressor = RandomForestRegressor()
            flat_tr_x = torch.reshape(tr_tensor_x, (tr_tensor_x.shape[0], tr_tensor_x.shape[1] * tr_tensor_x.shape[2])).cpu().numpy()
            flat_tr_y = tr_tensor_y.cpu().numpy()
            flat_tr_y = np.reshape(flat_tr_y, (flat_tr_y.shape[0]))
            rf_regressor.fit(flat_tr_x, flat_tr_y)
            shallow_RF_model = LineTracerShallowRFEnvironmentModel(rf_regressor, STATE_FEATURES, DEVICE)
            env_model = shallow_RF_model
            save_model(env_model, temp_model_path, model_prefix, 0)
            #save_best_model(env_model, best_model_dir_path + "best_rf_model.pt")

            best_models = [env_model]
            best_models_names = [model_prefix]
        # end model learning

        # begin model testing
        fot_length = te_np_log[0].shape[0]
        simul_length = fot_length - history_len
        init_history = te_tensor_x_init
        configs = te_tensor_c_init

        fot_maximum_displacements = get_maximum_displacements(te_tensor_log, COLOR_FEATURE_IDX, DENORM_LANE_KEEPING_COLOR_GOAL, scaler, history_len)
        fot_maximum_jerks = get_maximum_jerk(te_tensor_log, COLOR_FEATURE_IDX, history_len)
        fot_mean_jerks = get_average_jerk(te_tensor_log, COLOR_FEATURE_IDX, history_len)

        verification_report = []
        num_nondet_simul_repeat = 5

        if simul_deterministic_flag:
            verification_report.append(configs)
            verification_report.append(fot_maximum_displacements)
            verification_report.append(fot_maximum_jerks)
            verification_report.append(fot_mean_jerks)
        else:
            verification_report.append(configs.repeat(num_nondet_simul_repeat, 1))
            verification_report.append(fot_maximum_displacements.repeat(num_nondet_simul_repeat, 1))
            verification_report.append(fot_maximum_jerks.repeat(num_nondet_simul_repeat, 1))
            verification_report.append(fot_mean_jerks.repeat(num_nondet_simul_repeat, 1))

        verification_report_columns = []
        verification_report_columns.append('controller config')
        verification_report_columns.append('fot safety')
        verification_report_columns.append('fot comfort')
        verification_report_columns.append('fot mean comfort')

        for idx in range(len(best_models)):
            best_models[idx].eval()
            if simul_deterministic_flag:
                simul_log = deterministic_simulation(av, best_models[idx], simul_length, configs, init_history)
            else:
                simul_logs = []
                for simul_repeat_idx in range(num_nondet_simul_repeat):
                    simul_logs.append(non_deterministic_simulation(av, best_models[idx], simul_length, configs, init_history))
                simul_log = torch.concat(simul_logs, dim=0)

            #draw_plot(model_prefix + " testing", [te_tensor_log[0,:,[0]].cpu().numpy(), simul_log[0,:,[0]].cpu().numpy()], ["fot color", "simul color"])

            if idx == 0:
                validation_criteria_str = "loss"
            elif idx == 1:
                validation_criteria_str = "eucl"
            elif idx == 2:
                validation_criteria_str = "dtw"
            else:
                validation_criteria_str = "none"

            for te_config_name in te_config_idxs:
                simul_traces_name_for_vis.append(model_prefix + "_" + validation_criteria_str +"_config_"+ str(te_config_name))
            for te_vis_config_idx in range(len(te_config_idxs)):
                simul_traces_for_vis.append(simul_log[te_vis_config_idx * split_nums[2],:,[0]].cpu().numpy())

            simul_maximum_displacements = get_maximum_displacements(simul_log, COLOR_FEATURE_IDX, DENORM_LANE_KEEPING_COLOR_GOAL, scaler, history_len)
            simul_maximum_jerks = get_maximum_jerk(simul_log, COLOR_FEATURE_IDX, history_len)
            simul_mean_jerks = get_average_jerk(simul_log, COLOR_FEATURE_IDX, history_len)

            if simul_deterministic_flag:
                simul_eucls = get_euclidean(simul_log, te_tensor_log, COLOR_FEATURE_IDX + 1)
                simul_eucls = torch.reshape(simul_eucls, (simul_eucls.shape[0], 1))
                simul_dtws = get_dtw(simul_log, te_tensor_log, COLOR_FEATURE_IDX + 1)
                simul_dtws = torch.reshape(simul_dtws, (simul_dtws.shape[0], 1))
            else:
                simul_eucls = get_euclidean(simul_log, te_tensor_log.repeat(num_nondet_simul_repeat, 1, 1), COLOR_FEATURE_IDX + 1)
                simul_eucls = torch.reshape(simul_eucls, (simul_eucls.shape[0], 1))
                simul_dtws = get_dtw(simul_log, te_tensor_log.repeat(num_nondet_simul_repeat, 1, 1), COLOR_FEATURE_IDX + 1)
                simul_dtws = torch.reshape(simul_dtws, (simul_dtws.shape[0], 1))

            verification_report.append(simul_maximum_displacements)
            verification_report.append(simul_maximum_jerks)
            verification_report.append(simul_mean_jerks)
            verification_report.append(simul_eucls)
            verification_report.append(simul_dtws)
            verification_report_columns.append(best_models_names[idx] + ' safety')
            verification_report_columns.append(best_models_names[idx] + ' comfort')
            verification_report_columns.append(best_models_names[idx] + ' mean comfort')
            verification_report_columns.append(best_models_names[idx] + ' euclidean')
            verification_report_columns.append(best_models_names[idx] + ' dtw')

        verification_report = torch.stack(verification_report)
        verification_report = verification_report.cpu().numpy()
        verification_report = np.reshape(verification_report, (verification_report.shape[0], verification_report.shape[1]))
        report_df = pd.DataFrame(verification_report.T, columns=verification_report_columns, dtype=float)
        report_df.to_csv(verif_output_path + model_prefix + "_verif_report.csv")
        # end model testing

    simul_traces_for_vis = np.stack(simul_traces_for_vis)
    simul_traces_for_vis = np.reshape(simul_traces_for_vis, (simul_traces_for_vis.shape[0], simul_traces_for_vis.shape[1]))
    simul_traces_for_vis = simul_traces_for_vis.T
    for simul_name in simul_traces_name_for_vis:
        simul_traces_file.write(simul_name + ',')
    simul_traces_file.write('\n')

    for simulation_tick_trace in simul_traces_for_vis:
        for elem in simulation_tick_trace:
            simul_traces_file.write(str(elem) + ',')
        simul_traces_file.write('\n')

    simul_traces_file.close()
