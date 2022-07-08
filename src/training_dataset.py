class TrainingDataset:
    def __init__(self, tr_np_log, tr_tensor_log, tr_tensor_x, tr_tensor_y, tr_tensor_c, tr_tensor_x_init, tr_tensor_y_init, tr_tensor_c_init,
                 va_np_log, va_tensor_log, va_tensor_x, va_tensor_y, va_tensor_c, va_tensor_x_init, va_tensor_y_init, va_tensor_c_init):
        self.tr_np_log = tr_np_log
        self.tr_tensor_log = tr_tensor_log
        self.tr_tensor_x = tr_tensor_x
        self.tr_tensor_y = tr_tensor_y
        self.tr_tensor_c = tr_tensor_c
        self.tr_tensor_x_init = tr_tensor_x_init
        self.tr_tensor_y_init = tr_tensor_y_init
        self.tr_tensor_c_init = tr_tensor_c_init
        self.va_np_log = va_np_log
        self.va_tensor_log = va_tensor_log
        self.va_tensor_x = va_tensor_x
        self.va_tensor_y = va_tensor_y
        self.va_tensor_c = va_tensor_c
        self.va_tensor_x_init = va_tensor_x_init
        self.va_tensor_y_init = va_tensor_y_init
        self.va_tensor_c_init = va_tensor_c_init

