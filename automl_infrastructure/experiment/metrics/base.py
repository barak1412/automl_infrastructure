from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np


class Metric(ABC):
    def __init__(self, is_grouped=False, weighted=True):
        self._is_grouped = is_grouped
        self._weighted = weighted

    @abstractmethod
    def measure(self, y_true, classifier_prediction):
        pass


class SimpleMetric(Metric):

    def __init__(self, is_grouped=False, weighted=True):
        super().__init__(is_grouped=is_grouped, weighted=weighted)

    def measure(self, y_true, classifier_prediction):
        classes_measure = self.measure_lst(y_true, classifier_prediction)
        classes_measure = [v[1] for v in classes_measure.items()]
        if self._is_grouped:
            if self._weighted:
                classes_num = len(set(y_true + classifier_prediction.classes_pred))
                classes_size = SimpleMetric._calculate_classes_occurrences(y_true, classifier_prediction.classes_pred)
                weighted_sum = 0.0
                for i in range(len(classes_measure)):
                    if classes_num[i] > 0:
                        weighted_sum += classes_measure[i] / classes_size[i]
                weighted_sum = weighted_sum / classes_num
                return weighted_sum
            else:
                return np.mean(classes_measure)
        else:
            return classes_measure

    @abstractmethod
    def measure_lst(self, y_true, classifier_prediction):
        pass

    @staticmethod
    def _calculate_classes_occurrences(y_true, y_pred):
        classes_occur = {}
        joint_classes = y_true+y_pred
        for label in joint_classes:
            if label in classes_occur:
                classes_occur[label] += 1
            else:
                classes_occur[label] = 1
        return [v[1] for v in OrderedDict(sorted(classes_occur.items()))]
