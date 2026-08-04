from sklearn.impute import SimpleImputer ## handeling missing values
from sklearn.preprocessing import StandardScaler ## handeling feature scaling 
from sklearn.preprocessing import OrdinalEncoder ## ordinal encoding 

## piplines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import sys
from dataclasses import dataclass
import os
from src.exception import CustomerException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np

## data transformation cinfiguration 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')




## data ingestion config class

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info('Data Transformation initiated')
            categorical_cols=['cut','color','clarity']
            numerical_cols=['carat','depth','table','x','y','z']

            cut_categories=['Fair','Good','Very Good','Premium','Ideal']
            color_categories=['D','E','F','G','H','I','J']
            clarity_categories=['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initiated')

            ## numerical pipeline
            num_pipeline=Pipeline(
                
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            # categorical pipeline
            categorical_pipeline=Pipeline(
                
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            prepocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('cat_pipeline',categorical_pipeline,categorical_cols)
                ]
            )

            return prepocessor

            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomerException(e,sys)
        


    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')

            logging.info('Obtaining Preprocessor Object')

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name='price'
            drop_columns=[target_column_name,'id']

            input_feature_train_df=train_df.drop(columns=drop_columns)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns)
            target_feature_test_df=test_df[target_column_name]

            ## transforming using preprocessor obj
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying Preprocessing object on training and testing datasets.')

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info('Preprocessor pickle created and saved')

            return(
                
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Exception occured in the initiate_data_transformation')
            raise CustomerException(e,sys)
