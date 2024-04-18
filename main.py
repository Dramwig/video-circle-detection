from src.frame_processing import frame_processing
from src.data_generator import data_generator
from src.train import train
from src.test import test


if __name__ == '__main__':
    # for i in range(1, 6+1):
    #     frame_processing(str_index = str(i))
    # data_generator()
    train(num_epochs = 1000, batch_size = 50)
    for i in range(1, 6+1):
        test(str_index = str(i))
    
    
    # frame_processing()
    # data_generator()
    # train(num_epochs = 500)
    # test(orientations = 'y')