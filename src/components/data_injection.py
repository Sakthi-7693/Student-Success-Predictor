import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
from src.utils import save_object
from src.components.data_transformation import DataTransformation

#This code will create the artifacts folder and create a (train,test,data csv file)
@dataclass
class DataInjectionConfig:
    train_data_path : str=os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_path_data: str = os.path.join('artifacts', "data.csv")

class DataInjection:
    def __init__(self):
        self.injection_config = DataInjectionConfig()

    def initiate_data_injection(self):
        logging.info("Entered into data injection method")
        try:
            df = pd.read_csv(r"C:\Student-Success-Predictor\Model_Evaluation\StudentsPerformance.csv")
            logging.info("Sucessfully readed the dataset as dataframe")

            os.makedirs(os.path.dirname(self.injection_config.train_data_path), exist_ok=True)

            df.to_csv(self.injection_config.raw_path_data,index=False,header=True)
            logging.info("Train-test-split initiated")

            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)

            train_set.to_csv(self.injection_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.injection_config.test_data_path, index=False,header=True)

            logging.info("Injection of the data is completed")

            return(
                self.injection_config.train_data_path,
                self.injection_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
            
if __name__=="__main__":
    obj = DataInjection()
    train_data, test_data = obj.initiate_data_injection()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)