import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transform_object(self):
        try:
            num_col = ['reading score', 'writing score']
            cat_col = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_Pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler(with_mean=False))  # Added for sparse matrix compatibility
                ]
            )

            cat_Pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoder", OneHotEncoder())  # Removed StandardScaler from categorical pipeline
                ]
            )

            logging.info(f"Numerical columns : {num_col}")
            logging.info(f"Categorical columns : {cat_col}")

            preprocessor = ColumnTransformer(
                [
                    ("num_Pipeline", num_Pipeline, num_col),
                    ("cat_Pipeline", cat_Pipeline, cat_col)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transform_object()

            target_column_name = "math score"
            num_col = ['reading score', 'writing score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]  # Fixed syntax error here

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
