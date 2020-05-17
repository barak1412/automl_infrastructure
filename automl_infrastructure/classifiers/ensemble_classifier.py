from automl_infrastructure.classifiers import Classifier
import numpy as np


class EnsembleClassifier(Classifier):

    def __init__(self, name, input_models, ensemble_model, ensemble_extra_features=None):
        super().__init__(name)
        self._input_models = input_models
        self._ensemble_model = ensemble_model
        self._ensemble_features = ensemble_extra_features

    def _prepare_features(self, x):
        new_features = []
        for input_model in self._input_models:
            new_features.append(input_model.predict_proba(x))
        if self._ensemble_features is not None:
            new_features.append(x[self._ensemble_features])
        return np.concatenate(new_features, axis=1)

    def fit(self, x, y, **kwargs):
        # train input models
        for input_model in self._input_models:
            input_model.fit(x, y, **kwargs)

        # train ensemble model based on input models predictions
        final_x = self._prepare_features(x)
        self._ensemble_model.fit(final_x, y, **kwargs)

    def predict(self, x):
        final_x = self._prepare_features(x)
        return self._ensemble_model.predict(final_x)

    def predict_proba(self, x):
        final_x = self._prepare_features(x)
        return self._ensemble_model.predict_proba(final_x)

    def set_params(self, params):
        for input_model in self._input_models:
            if input_model.name in params:
                input_model.set_params(params[input_model.name])
        if self._ensemble_model.name in params:
            self._ensemble_model.set_params(params[self._ensemble_model.name])

    def get_params(self, deep=True):
        params = {self._ensemble_model.name: self._ensemble_model.get_params()}
        if deep:
            for input_model in self._input_models:
                params[input_model.name] = input_model.get_params()
        return params
