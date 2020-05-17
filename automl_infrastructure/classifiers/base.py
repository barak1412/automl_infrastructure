from abc import ABC, abstractmethod


class ClassifierPrediction(object):
    def __init__(self, y_pred, y_proba):
        self._y_pred = y_pred
        self._y_proba = y_proba

    @property
    def classes_pred(self):
        return self._y_pred

    @property
    def classes_proba(self):
        return self._y_proba


class Classifier(ABC):

    def __init__(self, name, features_cols=None):
        self._name = name
        self._features_cols = features_cols

    @property
    def name(self):
        return self._name

    @abstractmethod
    def fit(self, x, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

    @abstractmethod
    def get_params(self, deep=True):
        pass
