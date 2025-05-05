import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
import sys
import joblib

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, train_data_path):
        self.train_data_path = train_data_path
        self.data = None
        logger.info("Your Data Processing is initialized...")

    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            print(f"the data has successfully read from:{self.train_data_path}")
            print(self.data.shape)
            logger.info("Read the train and test data successfully")
            return self.data
        except Exception as e:
            logger.error(f"Error while reading data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def preprocess_data(self, df):
        try:
            df.drop_duplicates(inplace=True)
            logger.info("Data preprocessing completed for DataFrame")
            return df
        except Exception as e:
            logger.error(f"Error while preprocessing data: {e}")
            raise CustomException(str(e), sys.exc_info())


    def run(self):
        try:
            logger.info("Starting Data Processing Pipeline...")
            print("Starting Data Processing Pipeline...")
            self.data = self.load_data()
            print("self.data",self.data)
            self.data = self.preprocess_data(self.data)
            logger.info("End of Data Processing Pipeline...")
            return self.data
        except Exception as e:
            logger.error(f"Error in Data Processing Pipeline: {e}")
            raise CustomException(str(e), sys.exc_info())

if __name__ == "__main__":
    data_processor = DataProcessing(TRAIN_PATH)
    data_processor.run()