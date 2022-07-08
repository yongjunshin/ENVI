class EnviConfig:
    def __init__(self, envi_algo, lr, epoch,
                 bc_deterministic=None,
                 gail_x_bc=None,
                 gail_deterministic=None,
                 gail_algo=None,
                 gail_disc_learning_break=None,
                 gail_disc_iter=None,
                 gail_ppo_iter=None,
                 model_save_period=None):
        self.envi_algorithm = envi_algo
        self.lr = lr
        self.epoch = epoch

        self.bc_determinisitic = bc_deterministic

        self.gail_x_bc = gail_x_bc
        self.gail_deterministic = gail_deterministic
        self.gail_algo = gail_algo
        self.gaIl_disc_learning_break = gail_disc_learning_break
        self.gaIl_disc_iter = gail_disc_iter
        self.gail_ppo_iter = gail_ppo_iter

        self.model_save_period = model_save_period
