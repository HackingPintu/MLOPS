import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.model_dev import RandomForestRegressorModel
from .config import ModelNameConfig

@step
def train_model(
    X_train:pd.DataFrame,
    X_test:pd.DataFrame,
    y_train:pd.DataFrame,
    y_test:pd.DataFrame,
    config:ModelNameConfig,
)->RegressorMixin:
    try :
        model=None
        if config.model_name=="RandomForestRegressorModel":
            model=RandomForestRegressorModel()
            trained_model=model.train(X_train,y_train)
            return trained_model
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        raise
    
    