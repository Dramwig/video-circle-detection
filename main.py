from src.frame_processing import frame_processing
from src.data_generator import data_generator
from src.train import train
from src.test import test

import torch
import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Random seed {seed} has been set.')

if __name__ == '__main__':
    seed = 42  # 可以选择任何喜欢的数字作为种子
    seed_everything(seed)

    # for i in range(1, 6+1):
    #     frame_processing(str_index = str(i))
    # data_generator()
    # model_save_path = train(num_epochs = 1000, batch_size = 50)
    # for i in range(1, 6+1):
    #     test(str_index = str(i))
    
    # processed_data_paths = frame_processing()
    # data_generator(20000,2000,2000)
    # model_save_path = train(num_epochs = 500, if_use_scheduler = True)
    test(orientations = 'y', model_load_path = 'archive/seresnet_test_0.0071/seresnet_val_0.0061.pth')