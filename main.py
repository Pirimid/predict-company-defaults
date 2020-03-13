from src.generator import DataGenerator
from src.logger import LOGGER, setup_logger
import numpy as np


logger = setup_logger()

if __name__=="__main__":
    data = [np.random.rand(32,12), np.random.rand(32,12), np.random.rand(32,12), np.random.rand(32,12)]
    labels = [np.random.rand(1), np.random.rand(1), np.random.rand(1), np.random.rand(1)]
    timestamp = [np.random.rand(1), np.random.rand(1), np.random.rand(1), np.random.rand(1)]
    generator = DataGenerator(data, labels, timestamp)
    
    for i in range(2):
        logger.info("Got you the next step")
        print(generator.get_next_step()[0].shape)