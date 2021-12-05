from src.utils.common import read_config
from src.utils.data_management import get_data
from src.utils.model import get_model
import argparse
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename= os.path.join(log_dir,"running_logs.log"),level=logging.INFO, format=logging_str, filemode="a")


def training(config_path):

     config = read_config(config_path)
     
     validation_size =  config['params']['validation_size']
     Loss = config['params']['LOSS']
     metrics = config['params']['METRICS']
     optimizer = config['params']['OPTIMIZER']
     epochs = config['params']['EPOCHS']

     (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_size)
     logging.info("Mnist data loaded")

     model = get_model(Loss,optimizer,metrics)
     logging.info("Model Created")

     history = model.fit(X_train,y_train,validation_data = (X_valid, y_valid), epochs =epochs)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config.yaml")

    parsed_args = args.parse_args()
    logging.info(">>>>> starting training >>>>>")
    training(config_path=parsed_args.config)
    logging.info(">>>>> Ended training >>>>>") 

     