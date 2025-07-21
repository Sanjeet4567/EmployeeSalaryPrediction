import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models, params, cv=5, scoring='accuracy'):
    try:
        results = {}
        best_model = None
        best_score = 0
        best_model_name = None
        best_model_params = None

        for model_name, model in models.items():
            print(f"Training and tuning {model_name}...")
            param_grid = params[model_name]
            grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
            grid.fit(X_train, y_train)
            
            best_estimator = grid.best_estimator_
            y_pred = best_estimator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec=precision_score(y_test,y_pred)
            rec=recall_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred)
            roc=roc_auc_score(y_test,y_pred)
            
            results[model_name] = {
                'accuracy': acc,
                'precision':prec,
                'recall':rec,
                'f1 score':f1,
                'roc_auc_score':roc,
                'best_params': grid.best_params_,
                'model': best_estimator
            }

            if acc > best_score:
                best_score = acc
                best_model = best_estimator
                best_model_name = model_name
                best_model_params = grid.best_params_
        
        
        return {
            'all_results': results,
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_accuracy': best_score,
            'best_params': best_model_params
        }
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)