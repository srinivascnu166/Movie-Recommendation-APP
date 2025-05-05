import os
import joblib
import sys
import pandas as pd
import numpy as np
import pickle
import random
import scipy.sparse as sparse
from collections import namedtuple
from sklearn.model_selection import ParameterGrid
from rankfmc import RankFM
from rankfmc.evaluation import precision
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split, AUC_at_k
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from src.data_processing import DataProcessing
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import MODEL_PATH, MODEL_DIR, TRAIN_PATH


logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,DataProcessing = DataProcessing, train_data_path=TRAIN_PATH,  model_save_path=MODEL_DIR):
        
        self.model_save_path = model_save_path
        self.train_data_path = train_data_path
        self.model = None
        self.DataProcessing = DataProcessing(self.train_data_path)
        os.makedirs(self.model_save_path, exist_ok=True)
        logger.info("Model Training initialized .....")

    def load_data_from_data_processing(self):
        try:

            data = self.DataProcessing.run()
            return data
        except Exception as e:
            logger.error(f"Error while loading data from Redis: {e}")
            raise CustomException(str(e), sys.exc_info())

    def prepare_data(self):
        try:
            data = self.load_data_from_data_processing()
            #dictionary to store genres
            genre_dict = dict(zip(data.Movie_title, data.genre))

            interactions = data.iloc[:,:2]
            interactions.drop_duplicates(inplace=True)
            
            
            return interactions
        except Exception as e:
            logger.error(f"Error while preparing data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def hyperparameter_tuning_ALS(self, interactions):
        try:
            factors_list=[x for x in range(20, 50, 5)]
            epochs_list=[x for x in range(20, 50, 10)]
            learning_rate_list = [0.01]
            learning_schedule_list = ['constant', 'invscaling']
            K_list=[x for x in range(20, 50, 10)]
            search_type = 'random'
            n_iter=5
           
            random.seed(42)

            param_grid = {
                'factors': factors_list,
                'learning_rate':learning_rate_list,
                'epochs': epochs_list,
                'learning_schedule': learning_schedule_list,
                'k': K_list
            }

            all_params = list(ParameterGrid(param_grid))

            if search_type == 'random':
                sampled_params = random.sample(all_params, min(n_iter, len(all_params)))
            elif search_type == 'grid':
                sampled_params = all_params
            else:
                raise ValueError("search_type must be 'random' or 'grid'")

            best_score = -np.inf
            best_params = None

            np.random.seed(1492)
            interactions['random'] = np.random.random(size=len(interactions))
            test_pct = 0.25

            train_mask = interactions['random'] <  (1 - test_pct)
            valid_mask = interactions['random'] >= (1 - test_pct)

            interactions_train = interactions[train_mask][['User', 'Movie_title']]
            interactions_valid = interactions[valid_mask][['User', 'Movie_title']]


            for params in sampled_params:
                model = RankFM(factors=params['factors'], 
                           learning_rate=params['learning_rate'], learning_schedule=params['learning_schedule'])
                

                model.fit(interactions_train, epochs=params['epochs'], verbose=True)

                score = precision(model, interactions_valid, k = params['k'])

                if score > best_score:
                    best_score = score
                    best_params = params
     
            print(f"successfully tuned ALS model with accuracy of:{best_score}")
            
  
            return best_params, best_score
        except Exception as e:
            logger.error(f"Error while hyperparameter tuning ALS: {e}")
            raise CustomException(str(e), sys.exc_info())

    def train_and_evaluate(self):
        try:
            interactions = self.prepare_data()
            tuned_params , best_score = self.hyperparameter_tuning_ALS(interactions)
            keys_to_remove = {'k','epochs'}

            final_params = {k: v for k, v in tuned_params.items() if k not in keys_to_remove}
            model = RankFM(**final_params)

            model.fit(interactions.iloc[:,:2], epochs=tuned_params['epochs'], verbose=True)
            most_similar_items = model.similar_items('Paddington (2014)', n_items = 5) 
            with open(os.path.join(self.model_save_path, 'model.pkl'), 'wb') as out:
                pickle.dump(model, out)
                pickle.dump(interactions['Movie_title'].tolist(), out)
        except Exception as e:
            logger.error(f"Error while training and evaluating model: {e}")
            raise CustomException(str(e), sys.exc_info())

    
    def run(self):
        try:
            logger.info("Starting Model Training Pipeline...")
            print("Starting Model Training Pipeline...")
            interactions = self.prepare_data()
            best_params,best_score = self.hyperparameter_tuning_ALS(interactions)
            print(best_params,best_score)
            evaluate = self.train_and_evaluate()
            logger.info("Model Training is successful...")
            print("Model Training is successful...")
        except Exception as e:
            logger.error(f"Error in Model Training Pipeline: {e}")
            #self.experiment.log_other("error", str(e))
            raise CustomException(str(e), sys.exc_info())

if __name__ == "__main__":
    dataprocess = DataProcessing(TRAIN_PATH)
    model_trainer = ModelTraining(dataprocess)
    model_trainer.run()