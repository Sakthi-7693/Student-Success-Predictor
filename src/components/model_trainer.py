import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            x_train,y_train, x_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                 "Linear_Regression " : LinearRegression(),
                 "Decision_tree " : DecisionTreeRegressor(),
                 "Random_forest " : RandomForestRegressor(),
                 "KNeighborsRegressor" : KNeighborsRegressor(),
                 "XGBoost_Regression" : XGBRegressor(),
                 "AdaBoost_Regression " : AdaBoostRegressor(),
                 "CatBoost_Regression " : CatBoostRegressor(verbose=False),
                 "Gradient_Boosting_Regression " : GradientBoostingRegressor(),
                }   
            
            model_report :dict= evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test = y_test, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No bestmodel found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object (
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted =best_model.predict(x_test)
            r2_squared = r2_score(y_test,predicted)
            return r2_squared
        except Exception as e:
            raise CustomException(e, sys)