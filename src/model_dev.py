import logging
from abc import ABC,abstractmethod
from sklearn.ensemble import RandomForestRegressor

class Model(ABC):
    @abstractmethod
    def train(self,X_train,y_train):
        pass
    
class RandomForestRegressorModel(Model):
    def train(self, X_train, y_train,**kwargs):
        try:
            reg=RandomForestRegressor(**kwargs)
            reg.fit(X_train,y_train)
            return reg
        except Exception as e:
            logging.error("Error in training model: {}".format(e))
            raise e