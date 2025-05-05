from src.data_ingestion import DataIngestion
from config.paths_config import RAW_DIR, TRAIN_PATH, TEST_PATH
from src.configuration.mongo_db_connection import  MongoDBClient
from src.data_access.project_data import ProjectData
from src.data_processing import DataProcessing
# from src.model_training import ModelTraining
from src.rankfm_model_training import ModelTraining
from config.paths_config import MODEL_PATH, SCALER_PATH, MODEL_DIR, TRAIN_PATH
from pipeline.training_pipeline import TrainPipeline
# connection = MongoDBClient()
#data = ProjectData()
# pipline = DataIngestion(RAW_DIR)
# pipline.run()

# data_ingest = DataIngestion(RAW_DIR)
# data_ingest.run

trainer = TrainPipeline(RAW_DIR,TRAIN_PATH,MODEL_DIR)
trainer.run_pipeline()

# model_trainer = ModelTraining(DataProcessing,TRAIN_PATH,MODEL_DIR)

# model_trainer.run()
