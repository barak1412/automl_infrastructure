from automl_infrastructure.experiment.metrics.utils import parse_objective
from automl_infrastructure.classifiers import ClassifierPrediction
from sklearn.utils import check_random_state
import numpy as np


class PermutationImportance(object):
    def __init__(self, estimator, n_iter=3, scoring='accuracy', random_state=None):
        # initialize scoring function
        self._scoring = parse_objective(scoring)

        self._n_iter = n_iter
        self._rng = np.random.RandomState(seed=random_state)
        self._estimator = estimator

    def fit(self, X, y):
        # calculate base scoring
        base_score = self._calculate_score(X, y)

        scores_decreases = {}
        for feature in X:
            scores_decreases[feature] = self._get_scores_shuffled(X, y, feature)
            scores_decreases[feature] = [base_score - score for score in scores_decreases[feature]]
        print(scores_decreases)

    def _calculate_score(self, X, y):
        pred_y = self._estimator.predict(X)
        proba_y = self._estimator.predict(X)
        estimator_prediction = ClassifierPrediction(pred_y, proba_y)
        scoring = self._scoring(y, estimator_prediction)
        return scoring

    def _get_scores_shuffled(self, X, y, feature):
        shuffles_generator = self._iter_shuffled(X, feature)
        return np.array([self._calculate_score(X_shuffled, y) for X_shuffled in shuffles_generator])

    def _iter_shuffled(self, X, feature):
        X_res = X.copy()
        for i in range(self._n_iter):
            X_res[feature] = self._rng.permutation(X[feature].values)
            yield X_res
            X_res[feature] = X[feature]
