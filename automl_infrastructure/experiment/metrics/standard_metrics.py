from automl_infrastructure.experiment.metrics import Metric
from sklearn.metrics import f1_score, precision_score, recall_score, cohen_kappa_score
from collections import OrderedDict


class F1Score(Metric):
    def __init__(self, is_grouped=False, weighted=True):
        super().__init__(is_grouped, weighted)

    def measure(self, y_true, classifier_prediction):
        if self._is_grouped:
            if self._weighted:
                return f1_score(y_true, classifier_prediction.classes_pred, average='weighted')
            else:
                return f1_score(y_true, classifier_prediction.classes_pred, average='micro')
        else:
            return f1_score(y_true, classifier_prediction.classes_pred, average=None)


class Precision(Metric):
    def __init__(self, is_grouped=False, weighted=True):
        super().__init__(is_grouped, weighted)

    def measure(self, y_true, classifier_prediction):
        if self._is_grouped:
            if self._weighted:
                return precision_score(y_true, classifier_prediction.classes_pred, average='weighted')
            else:
                return precision_score(y_true, classifier_prediction.classes_pred, average='micro')
        else:
            return precision_score(y_true, classifier_prediction.classes_pred, average=None)


class Recall(Metric):
    def __init__(self, is_grouped=False, weighted=True):
        super().__init__(is_grouped, weighted)

    def measure(self, y_true, classifier_prediction):
        if self._is_grouped:
            if self._weighted:
                return recall_score(y_true, classifier_prediction.classes_pred, average='weighted')
            else:
                return recall_score(y_true, classifier_prediction.classes_pred, average='micro')
        else:
            return recall_score(y_true, classifier_prediction.classes_pred, average=None)


class CohenKappa(Metric):
    def __init__(self, is_grouped=True, weighted=True, linear=True):
        super().__init__(is_grouped, weighted)
        self._linear = linear
        if not is_grouped:
            raise Exception('CohenKappa metric must be grouped: is_grouped should be True.')

    def measure(self, y_true, classifier_prediction):
        if self._weighted:
            if self._linear:
                weights = 'linear'
            else:
                weights = 'quadratic'
            return cohen_kappa_score(y_true, classifier_prediction.classes_pred, weights=weights)
        else:
            return cohen_kappa_score(y_true, classifier_prediction.classes_pred, weights=None)


class Support(Metric):
    def __init__(self, is_grouped=False, weighted=True):
        super().__init__(is_grouped, weighted)

    def measure(self, y_true, classifier_prediction):
        classes_occur_dict = Support._calculate_classes_occurrences(y_true)
        classes_occur = [v for v in classes_occur_dict.values()]
        if self._is_grouped:
            return sum(classes_occur)
        else:
            return classes_occur

    @staticmethod
    def _calculate_classes_occurrences(y_true):
        classes_occur = {}
        for label in y_true:
            if label in classes_occur:
                classes_occur[label] += 1
            else:
                classes_occur[label] = 1
        return OrderedDict(sorted(classes_occur.items()))


class MetricFactory(object):
    standard_metrics = {
        'f1_score': F1Score(is_grouped=False),
        'precision': Precision(is_grouped=False),
        'recall': Recall(is_grouped=False),
        'support': Support(is_grouped=False)
    }

    @staticmethod
    def create(name):
        if name not in MetricFactory.standard_metrics:
            raise Exception('Metric named {} is not supported'.format(name))
        return MetricFactory.standard_metrics[name]


class ObjectiveFactory(object):

    standard_objectives = {
        'f1_score': F1Score(is_grouped=True, weighted=False),
        'weighted_f1_score': F1Score(is_grouped=True, weighted=True),
        'precision': Precision(is_grouped=True, weighted=False),
        'weighted_precision': Precision(is_grouped=True, weighted=True),
        'recall': Recall(is_grouped=True, weighted=False),
        'weighted_recall': Recall(is_grouped=True, weighted=True),
        'linear_cohen_kappa': CohenKappa(is_grouped=True, weighted=True, linear=True),
        'quadratic_cohen_kappa': CohenKappa(is_grouped=True, weighted=True, linear=False),
        'cohen_kappa': CohenKappa(is_grouped=True, weighted=False)
    }

    @staticmethod
    def create(name):
        if name not in ObjectiveFactory.standard_objectives:
            raise Exception('Objective named {} is not supported'.format(name))
        return ObjectiveFactory.standard_objectives[name]


