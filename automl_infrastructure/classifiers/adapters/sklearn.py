from automl_infrastructure.classifiers import Classifier
from sklearn.preprocessing import LabelEncoder


class SklearnClassifierAdapter(Classifier):

    def __init__(self, name, sklearn_model, features_cols=None, encode_labels=False):
        super().__init__(name, features_cols=features_cols)
        self._sklearn_model = sklearn_model
        self._encode_labels = encode_labels
        if encode_labels:
            self._label_encoder = LabelEncoder()

    def _get_effective_x(self, x):
        if self._features_cols is not None:
            effective_x = x[self._features_cols]
        else:
            effective_x = x
        return effective_x

    def fit(self, x, y, **kwargs):
        x = self._get_effective_x(x)
        if self._encode_labels:
            self._label_encoder.fit(y)
            y = self._label_encoder.transform(y)
        self._sklearn_model.fit(x, y, **kwargs)

    def predict(self, x):
        x = self._get_effective_x(x)
        prediction_df = self._sklearn_model.predict(x)
        if self._encode_labels:
            prediction_df = self._label_encoder.inverse_transform(prediction_df)
        return prediction_df

    def predict_proba(self, x):
        x = self._get_effective_x(x)
        prediction_df = self._sklearn_model.predict_proba(x)
        return prediction_df

    def set_params(self, params):
        self._sklearn_model.set_params(**params)

    def get_params(self, deep=True):
        return self._sklearn_model.get_params(deep=deep)

