from src.frame_processing import frame_processing
from src.data_generator import data_generator
from src.train import train


if __name__ == '__main__':
    dir_processed_data = frame_processing()
    data_generator()
    train()