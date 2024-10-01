import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin
from src.model_dev import RandomForestRegressorModel

@step
def train_model(
    X_train:pd.DataFrame
    X_test:pd.DataFrame,
    y_train:pd.DataFrame,
    y_test:pd.DataFrame,
)->RegressorMixin:
    
    
    