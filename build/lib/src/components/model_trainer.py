# Basic Import
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'LogisticRegression': LogisticRegression(max_iter=1500),
                'NaiveBayes': GaussianNB(),
                'RandomForest': RandomForestClassifier(),
                'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
                'KNN': KNeighborsClassifier()
            }

            params = {
                'LogisticRegression': {
                    'C': [0.001,0.01,0.1, 1, 5, 10],
                    'solver': ['lbfgs', 'liblinear']
                },
                'NaiveBayes': {},  # No tuning needed for GaussianNB (can be left empty)
                'RandomForest': {
                    'n_estimators': [40,50,60,80,100],
                    'max_depth': [None,3,5,7]
                },
                'XGBoost': {
                    'n_estimators': [30,40,50,60,80,100],
                    'max_depth': [3,5,7],
                    'learning_rate': [0.01,0.05,0.1]
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7, 9, 11, 13],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            }
            results:dict = evaluate_model(X_train, y_train, X_test, y_test, models, params)
            model_report=pd.DataFrame(results["all_results"]).T
            model_report=model_report.sort_values("accuracy",ascending=False)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : ')

            # To get best model score from dictionary 

            best_model_name = results["best_model_name"]
            
            best_model = results["best_model"]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {results["best_accuracy"]}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {results["best_accuracy"]}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)