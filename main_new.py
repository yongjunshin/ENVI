from src.av_controller import AV
from src.av_env_model import AVEnvModelDNN
from src.bc import BCTrainer
from src.data_manager import *
from src.gail import GAILTrainer
from src.gail_actor_critic import GAILActorCriticTrainer
from src.gail_reinforce import GAILReinforceTrainer
from src.data_manager_lk_fot import load_tmvk_norm_np_data, load_tmvu_norm_np_data, np_log_to_tensor_training_data
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.gail_tdac import GAILTdacTrainer
from src.lk_controller import LaneKeepingSystem
from src.shallow_model import LineTracerShallowRFEnvironmentModel

# fixed setting
from src.training_dataset import TrainingDataset

FOT_LOG_DIR = "data/AV-FOT/"
TR_FLAG = 0
VA_FLAG = 1
TE_FLAG = 2
COLUMNS = [' color', ' distance', ' angle', ' speed']
STATE_FEATURES = 2
ACTION_FEATURES = 2
FC_LAYER_FEATURES = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# user setting
x_configs = [0.4, 0.5, 0.6, 0.7, 0.8]
y_configs = [0.6, 0.9, 1.2, 1.5, 1.8]
z_configs = [180]
tr_config_idxs = [0, 2, 4]
va_config_idxs = [0, 2, 4]
te_config_idxs = [1, 3]
split_nums = [10, 10, 30]
history_len = 10
algorithm = 'bc'
lr = 0.00005
epoch = 10


configs = []
for x in x_configs:
    for y in y_configs:
        for z in z_configs:
            configs.append([x, y, z])

configs_string = []
for config in configs:
    configs_string.append(str(config[0])+"_"+str(config[1])+"_"+str(config[2]))

np_data, scaler = prepare_np_fot_data(FOT_LOG_DIR, configs_string, COLUMNS, split_nums)

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

av = AV(scaler)
env_model = AVEnvModelDNN(STATE_FEATURES + ACTION_FEATURES, history_len, FC_LAYER_FEATURES, STATE_FEATURES, DEVICE)



if algorithm == 'bc':
    # BC
    bc = BCTrainer(DEVICE, av, lr)
    losses, va_losses = bc.train(env_model, epoch, tr_dataset)
    print(losses)
    print(va_losses)
    plt.figure(figsize=(9, 3))
    plt.plot(losses, label="tr_loss")
    plt.plot(va_losses, label="va_loss")
    plt.legend()
    plt.show()
elif algorithm == 'gail':
    # GAIL
    gail = GAILReinforceTrainer(DEVICE, av, STATE_FEATURES, ACTION_FEATURES, history_len, lr)
    reward_list = gail.train(env_model, epoch, tr_tensor_x, tr_tensor_y, tr_tensor_c, tr_tensor_x_init, tr_tensor_c_init, tr_tensor_log.shape[1] - history_len, tr_tensor_log)
    plt.figure(figsize=(10, 5))
    plt.plot(np.array(reward_list), label="Model reward", color="orange")
    plt.legend()
    plt.show()


# from sklearn.ensemble import RandomForestRegressor
# rf_regressor = RandomForestRegressor()
# flat_tr_x = torch.reshape(tr_x, (tr_x.shape[0], tr_x.shape[1] * tr_x.shape[2])).cpu()
# flat_tr_y = tr_y.cpu()
# rf_regressor.fit(flat_tr_x, flat_tr_y)
# shallow_RF_model = LineTracerShallowRFEnvironmentModel(rf_regressor, device)
# env_model = shallow_RF_model


simul_length = te_np_log[0].shape[0]
history = te_tensor_x_init
configs = te_tensor_c_init
simul_log = history
env_model.eval()
av.reset()
print("simulation start")
for i in range(simul_length - history_len):
    # original data
    # state = te_tensor_log[:, i+history_len, :STATE_FEATURES]

    # deterministic simulation
    # state = env_model(history).detach()

    # non-deterministic simulation
    dist = env_model.get_distribution(history)
    state = dist.sample().detach()

    action = av.act_parallel(history, state, configs)

    new_step = torch.cat((state, action), dim=1)
    new_step = torch.reshape(new_step, (new_step.shape[0], 1, new_step.shape[1]))
    simul_log = torch.cat((simul_log, new_step), dim=1)
    history = simul_log[:, -history_len:, :]

print("visualization start")
for feature_idx in range(2):
    print("vis")
    plt.figure(figsize=(10, 5))
    plt.plot(te_np_log[0][:, [feature_idx]], label="fot")
    plt.plot(simul_log[0, :, [feature_idx]].cpu().detach().numpy(), label="simul")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(te_np_log[11][:, [feature_idx]], label="fot")
    plt.plot(simul_log[11, :, [feature_idx]].cpu().detach().numpy(), label="simul")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(te_np_log[22][:, [feature_idx]], label="fot")
    plt.plot(simul_log[22, :, [feature_idx]].cpu().detach().numpy(), label="simul")
    plt.legend()
    plt.show()
print("visualization end")