import logging 
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass
    
    
class DataPreprocessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            label_encoder=LabelEncoder()
            data['job_title_encoded']=label_encoder.fit_transform(data['job_title'])
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
    
class DataDivideStartegy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame ,pd.Series]:
        try:
            X=data[['certification','entity_id','job_title_encoded']]
            y=data['rate']
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error("Error in dividing data :{}".format(e))
            raise
        
class DataCleaning:
    def __init__(self,data:pd.DataFrame,startegy:DataStrategy):
        self.data=data
        self.startegy=startegy
        
    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        try:
            return self.startegy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e