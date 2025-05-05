import sys
import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
#from src.constants import DATABASE_NAME, MONGODB_URL_KEY
from config.paths_config import RAW_DIR, TRAIN_PATH, TEST_PATH
from src.data_access.project_data import ProjectData
from src.constants import DATABASE_NAME, COLLECTION_NAME

#import kaggle
#from comet_ml import Experiment  # Import Comet ML

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        

    def download_dataset(self):
        try:
            logger.info("loading data from MongoDB")
            proj_data = ProjectData()
            data = proj_data.export_collection_as_dataframe(COLLECTION_NAME)
            print("COLLECTION_NAME---------->>>",COLLECTION_NAME)
            logger.info(f"successfully loaded data with rows:{data.shape[0]} and columns:{data.shape[1]}")
            dataset = "movie lens"
            logger.info(f"Dataset {dataset} downloaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading dataset from MongoDB: {e}")
            raise CustomException(str(e), sys)

    def save_data(self, data):
        try:
            data.to_csv(TRAIN_PATH, index=False)
            logger.info(f"Data saved successfully. Train rows: {len(data)}")
        except Exception as e:
            logger.error(f"Error while saving data: {e}")
            raise CustomException(str(e), sys)

    def run(self):
        try:
            logger.info("Data Ingestion Pipeline Started...")
            train_df = self.download_dataset()
            self.save_data(train_df)
            logger.info("End of Data Ingestion Pipeline")
        except Exception as e:
            logger.error(f"Error in Data Ingestion Pipeline: {e}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion(RAW_DIR)
    data_ingestion.run()