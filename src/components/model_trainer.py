import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model 

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):
        try:
            logging.info("splitting the training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                )
            # Specify hyperparameters, this is what we did but the performance is same w.r.t without specifying any hyperparameter. 
            linear_regression_params = {}
            k_neighbors_params = {'n_neighbors': 7}
            decision_tree_params = {'max_depth': 10, 'min_samples_split': 3}
            gradient_boosting_params = {'n_estimators': 200, 'learning_rate': 0.1}
            random_forest_params = {'n_estimators': 200, 'max_depth': None}
            xgb_regressor_params = {'n_estimators': 200, 'learning_rate': 0.1}
            catboost_regressor_params = {'iterations': 200, 'learning_rate': 0.2, 'verbose': False}
            adaboost_regressor_params = {'n_estimators': 100, 'learning_rate': 0.1}

# Create model instances with specified hyperparameters
            models = {
                        "Linear Regression": LinearRegression(**linear_regression_params),
                        "K-Neighbors Regressor": KNeighborsRegressor(**k_neighbors_params),
                        "Decision Tree": DecisionTreeRegressor(**decision_tree_params),
                        "Gradient Boosting": GradientBoostingRegressor(**gradient_boosting_params),
                        "Random Forest Regressor": RandomForestRegressor(**random_forest_params),
                        "XGBRegressor": XGBRegressor(**xgb_regressor_params),
                        "CatBoosting Regressor": CatBoostRegressor(**catboost_regressor_params),
                        "AdaBoost Regressor": AdaBoostRegressor(**adaboost_regressor_params)
                     }
            #models = {
                 #"Linear Regression": LinearRegression(),
                 #"K-Neighbors Regressor": KNeighborsRegressor(),
                 #"Decision Tree": DecisionTreeRegressor(),
                 #"Gradient Boosting": GradientBoostingRegressor(),
                 #"Random Forest Regressor": RandomForestRegressor(),
                 #"XGBRegressor": XGBRegressor(), 
                 #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                 #"AdaBoost Regressor": AdaBoostRegressor()
                   # }
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train, X_test=X_test, y_test=y_test,models=models)
            
            #to get the best model score from dict
            best_model_score=max(sorted(model_report.values()))

            # Get the best model name
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model=models[best_model_name]
            print("best model is:",best_model_name)
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"best model found on both the training and testing datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model)
            
            predicted=best_model.predict(X_test)
            r2_score_value=r2_score(y_test,predicted)
            return r2_score_value


        except Exception as e:
            raise CustomException(e,sys)  
    
     