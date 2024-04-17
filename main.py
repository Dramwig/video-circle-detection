from src.frame_processing import frame_processing
from src.data_generator import data_generator
from src.train import train
from src.test import test


if __name__ == '__main__':
    #dir_processed_data = frame_processing()
    #data_generator()
    #train()
    #test()
    for i in range(1, 6):
        test(str_index = str(i))