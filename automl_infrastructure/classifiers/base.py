from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from automl_infrastructure.utils import random_str


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
        self._vector_features_mapping = {}

    @staticmethod
    def _is_feature_list_type(df, feature):
        # for empty DataFrame or non object column should be False
        if df.shape[0] == 0 or df[feature].dtype != 'object':
            return False

        # for non-empty DataFrame we should check for list or numpy array type
        list_supported_types = [list, np.array, np.ndarray]
        for supported_type in list_supported_types:
            if (df[feature].apply(type) == supported_type).all():
                return True
        return False

    def _unroll_list_feature(self, df, feature):
        if feature in self._vector_features_mapping:
            new_column_names = self._vector_features_mapping[feature]
        else:
            # retrieve feature vector length, assuming DataFrame isn't empty
            vector_dim = len(df[feature][0])

            # generate columns
            random_postfix = random_str(5)
            new_column_names = ['{}_{}_{}'.format(feature, random_postfix, i) for i in range(vector_dim)]
            self._vector_features_mapping[feature] = new_column_names

        df[new_column_names] = pd.DataFrame(df[feature].tolist(), index=df.index)
        del df[feature]

    def _get_effective_x(self, x, reset_features_mapping=False):
        if self._features_cols is not None:
            # narrow down features
            effective_x = x[self._features_cols]
        else:
            effective_x = x
        if reset_features_mapping:
            self._vector_features_mapping.clear()

        # unroll list type (a.k.a vector type) features to several features
        features_lst = [c for c in effective_x]
        for feature in features_lst:
            if BasicClassifier._is_feature_list_type(effective_x, feature):
                self._unroll_list_feature(effective_x, feature)
        return effective_x

    def fit(self, x, y, **kwargs):
        x = self._get_effective_x(x, reset_features_mapping=True)
        self._fit(x, y, **kwargs)

    def predict(self, x):
        x = self._get_effective_x(x, reset_features_mapping=False)
        return self._predict(x)

    def predict_proba(self, x):
        x = self._get_effective_x(x, reset_features_mapping=False)
        return self._predict_proba(x)

    def _predict(self, x):
        pass

    def _predict_proba(self, x):
        pass

    def _fit(self, x, y, **kwargs):
        pass

