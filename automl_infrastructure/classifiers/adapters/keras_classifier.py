from automl_infrastructure.classifiers import BasicClassifier
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class KerasClassifierAdapter(BasicClassifier):

    def __init__(self, name, keras_classifier, features_cols=None):
        super().__init__(name, features_cols=features_cols)
        self._keras_classifier = keras_classifier
        self._label_binarizer = LabelBinarizer()

    def _fit(self, x, y, **kwargs):
        self._label_binarizer.fit(y)
        y_binarized = self._label_binarizer.transform(y)
        self._keras_classifier.fit(x, y_binarized, **kwargs)

    def _predict(self, x):
        prediction_df = self._keras_classifier.predict(x)

        # transform vectors predictions to labels
        prediction_df = np.array(list(map(lambda i: self._label_binarizer.classes_[i], prediction_df)))
        return prediction_df

    def _predict_proba(self, x):
        prediction_df = self._keras_classifier.predict_proba(x)
        return prediction_df

    def set_params(self, params):
        self._keras_classifier.set_params(**params)

    def get_params(self, deep=True):
        return self._keras_classifier.get_params(deep=deep)

