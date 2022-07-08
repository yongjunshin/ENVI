from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.envi_util import save_model, evaluate_model
from src.simulation import *


class BCTrainer:
    def __init__(self, name, device, sut, lr):
        self.name = name
        self.device = device
        self.sut = sut
        self.loss_fn = torch.nn.MSELoss()
        self.lr = lr

    def train(self, model, epochs, tr_dataset, model_dir_path, deterministic_flag=False, model_save_period=1):
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        tr_losses = []
        va_losses = []
        tr_euclidean_losses = []
        va_euclidean_losses = []
        tr_dtw_losses = []
        va_dtw_losses = []

        model_save_epochs = []
        for epch in tqdm(range(epochs), desc="Training"):

            self.update_model(model, optimiser, tr_dataset, deterministic_flag)

            if epch % model_save_period == 0:
                model_save_epochs.append(epch)

                tr_loss, va_loss, tr_euclidean_loss, va_euclidean_loss, tr_dtw_loss, va_dtw_loss = \
                    evaluate_model(self, model, deterministic_flag, tr_dataset, self.name)

                tr_losses.append(tr_loss)
                va_losses.append(va_loss)
                tr_euclidean_losses.append(tr_euclidean_loss)
                va_euclidean_losses.append(va_euclidean_loss)
                tr_dtw_losses.append(tr_dtw_loss)
                va_dtw_losses.append(va_dtw_loss)

                save_model(model, model_dir_path, self.name, epch)

        return model_save_epochs, tr_losses, va_losses, tr_euclidean_losses, va_euclidean_losses, tr_dtw_losses, va_dtw_losses

    def update_model(self, model, optimizer, tr_dataset, deterministic_flag):
        model.train()
        tr_x = tr_dataset.tr_tensor_x
        tr_y = tr_dataset.tr_tensor_y
        tr_dl = DataLoader(dataset=TensorDataset(tr_x, tr_y), batch_size=512, shuffle=True)

        for _, (x_batch, y_batch) in enumerate(tr_dl):
            if deterministic_flag:
                # deterministic
                y_pred = model(x_batch)
                loss = self.loss_fn(y_batch, y_pred)
            else:
                # non-deterministic
                bc_dist = model.get_distribution(x_batch)
                loss = -bc_dist.log_prob(y_batch)
                loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # def evaluate_model_loss(self, model, eval_data, deterministic_flag):
    #     model.eval()
    #     x = eval_data[0]
    #     y = eval_data[1]
    #     dl = DataLoader(dataset=TensorDataset(x, y), batch_size=512, shuffle=True)
    #
    #     loss_acum = 0
    #     iter = 0
    #     for i, (x_batch, y_batch) in enumerate(dl):
    #         if deterministic_flag:
    #             # deterministic
    #             y_pred = model(x_batch)
    #             loss = self.loss_fn(y_batch, y_pred)
    #             loss_acum = loss_acum + loss.item()
    #         else:
    #             # non-deterministic
    #             bc_dist = model.get_distribution(x_batch)
    #             loss = -bc_dist.log_prob(y_batch)
    #             loss = loss.mean()
    #             loss_acum = loss_acum + loss.item()
    #         iter = iter + 1
    #
    #     return loss_acum / iter
