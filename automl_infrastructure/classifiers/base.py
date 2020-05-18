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


class BasicClassifier(Classifier, ABC):

    def __init__(self, name, features_cols=None):
        super().__init__(name, features_cols=features_cols)

    def _get_effective_x(self, x):
        if self._features_cols is not None:
            effective_x = x[self._features_cols]
        else:
            effective_x = x
        return effective_x

    def fit(self, x, y, **kwargs):
        x = self._get_effective_x(x)
        self._fit(x, y, **kwargs)

    def predict(self, x):
        x = self._get_effective_x(x)
        return self._predict(x)

    def predict_proba(self, x):
        x = self._get_effective_x(x)
        return self._predict_proba(x)

    def _predict(self, x):
        pass

    def _predict_proba(self, x):
        pass

    def _fit(self, x, y, **kwargs):
        pass

