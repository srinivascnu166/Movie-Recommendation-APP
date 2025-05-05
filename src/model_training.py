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
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import train_test_split, AUC_at_k
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from src.data_processing import DataProcessing
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import MODEL_PATH, MODEL_DIR, TRAIN_PATH

# from comet_ml import Experiment  

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
            data['User'] = data['User'].astype(str)
            data['Movie_title'] = data['Movie_title'].astype(str)
            data['Movie_title'] = data['Movie_title'].str.strip()

            #get category codes
            data['User'] = data['User'].astype("category")
            data['Movie_title'] = data['Movie_title'].astype("category")
            #creating unique ID's for users and items
            data['user_id'] = data['User'].cat.codes
            data['item_id'] = data['Movie_title'].cat.codes
                    
            # creating Mappings and storing into dictionaries............
            userID_to_UID = dict(enumerate(data['User'].cat.categories))
            UID_to_userID = dict(map(reversed,userID_to_UID.items()))
                    
            itemID_to_ItemNumber = dict(enumerate(data['Movie_title'].cat.categories))
            ItemNumber_to_itemID = dict(map(reversed,itemID_to_ItemNumber.items())) 
 
            #creating sparse matrices.....
            # The implicit library expects data as a item-user matrix
            sparse_item_user = sparse.csr_matrix((data['rating'].astype(float), (data['user_id'], data['item_id'])))

            print("data preparation successfully done...!!!")  
            DictionaryArtifacts = namedtuple("DictionaryArtifacts", ["genre_dict", "userID_to_UID", "UID_to_userID", "itemID_to_ItemNumber", "ItemNumber_to_itemID"])
       
            dictionary_artifacts = [genre_dict, userID_to_UID, UID_to_userID, itemID_to_ItemNumber, ItemNumber_to_itemID]
            return dictionary_artifacts , sparse_item_user
        except Exception as e:
            logger.error(f"Error while preparing data: {e}")
            raise CustomException(str(e), sys.exc_info())

    def hyperparameter_tuning_ALS(self, user_items):
        try:
            factors_list=[x for x in range(20, 200, 5)]
            iterations_list=[x for x in range(20, 200, 10)]
            alpha_list=[10, 20, 40]
            K_list=[x for x in range(20, 50, 10)]
            search_type = 'random'
            n_iter=5

            random.seed(42)

            param_grid = {
                'factors': factors_list,
                'iterations': iterations_list,
                'alpha_val': alpha_list,
                'K': K_list
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

            for params in sampled_params:
                model = AlternatingLeastSquares(factors=params['factors'], 
                                                iterations=params['iterations'])
                train, test = train_test_split(user_items, train_percentage = 0.3)
                print(type(train),type(test))
                model.fit(train*params['alpha_val'])
                print("-------->params['K']",params['K'], type(params['K']))
                print(user_items.get_shape())
                score = AUC_at_k(model, train, test, K=params['K'], show_progress=True)
                print(score.dtype,score)

                if score > best_score:
                    best_score = score
                    best_params = params
     
            return best_params, best_score
        except Exception as e:
            logger.error(f"Error while hyperparameter tuning ALS: {e}")
            raise CustomException(str(e), sys.exc_info())

    def train_and_evaluate(self, sparse_item_user):
        try:
            dictionary_artifacts,sparse_item_user = self.prepare_data()
            keys_to_remove = {'K', 'alpha_val'}
            final_params = {
                'factors': 20,
                'iterations': 40,
                    }
            model = AlternatingLeastSquares(**final_params)
            weighted_user_items = (sparse_item_user * 40.0).astype('double')

            model.fit(weighted_user_items)

            with open(os.path.join(self.model_save_path, 'model.pkl'), 'wb') as out:
                pickle.dump(model, out)
                pickle.dump(sparse_item_user, out)
                pickle.dump(dictionary_artifacts,out)
            print(f"model saved successfully at:{self.model_save_path}")
        except Exception as e:
            logger.error(f"Error while training and evaluating model: {e}")
            raise CustomException(str(e), sys.exc_info())

    
    def run(self):
        try:
            logger.info("Starting Model Training Pipeline...")
            print("Starting Model Training Pipeline...")
            _, user_item_matrix = self.prepare_data()
            evaluate = self.train_and_evaluate(user_item_matrix)
            logger.info("Model Training is successful...")
            print("Model Training is successful...")
           
        except Exception as e:
            logger.error(f"Error in Model Training Pipeline: {e}")
            raise CustomException(str(e), sys.exc_info())

if __name__ == "__main__":
    dataprocess = DataProcessing(TRAIN_PATH)
    model_trainer = ModelTraining(dataprocess)
    model_trainer.run()