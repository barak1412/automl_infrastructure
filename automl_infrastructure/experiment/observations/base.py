from abc import ABC, abstractmethod
from automl_infrastructure.experiment.metrics import MetricFactory
from automl_infrastructure.experiment.metrics import Metric
import pandas as pd


class Observation(ABC):
    def __init__(self, metric):
        if isinstance(metric, str):
            metric_obj = MetricFactory.create(metric)
            self._metric_func = lambda y_true, classifier_prediction: metric_obj.measure(y_true, classifier_prediction)
        elif isinstance(metric, Metric):
            self._metric_func = lambda y_true, classifier_prediction: metric.measure(y_true, classifier_prediction)
        elif callable(metric):
            self._metric_func = metric
        else:
            raise Exception('Unsupported metric type {} for observation.'.format(type(metric)))

    @abstractmethod
    def observe(self, y_true_lst, classifier_prediction_lst, output_class_col='CLASS', output_observation_col='OBSERVATION'):
        pass


class SimpleObservation(Observation):
    def __init__(self, metric):
        super().__init__(metric)

    @abstractmethod
    def agg_func(self, values):
        pass

    def observe(self, y_true_lst, classifier_prediction_lst, output_class_col='CLASS', output_observation_col='OBSERVATION'):
        # extract all unique classes names
        unique_classes_names = []
        for j in range(len(classifier_prediction_lst)):
            for i in range(len(classifier_prediction_lst[j].classes_pred)):
                if classifier_prediction_lst[j].classes_pred[i] not in unique_classes_names:
                    unique_classes_names.append(classifier_prediction_lst[j].classes_pred[i])
                if y_true_lst[j][i] not in unique_classes_names:
                    unique_classes_names.append(y_true_lst[j][i])
        unique_classes_names = sorted(unique_classes_names)

        # prepare dictionary of classes and their metric values
        classes_observations_dict = {}
        for class_label in unique_classes_names:
            classes_observations_dict[class_label] = []

        # collect values for each class
        for classifier_prediction, y_true in zip(classifier_prediction_lst, y_true_lst):
            metric_values = self._metric_func(y_true, classifier_prediction)
            for i in range(len(metric_values)):
                classes_observations_dict[unique_classes_names[i]].append(metric_values[i])

        # aggregate values for each class
        for class_label in classes_observations_dict:
            classes_observations_dict[class_label] = self.agg_func(classes_observations_dict[class_label])

        classes_col = []
        observation_col = []
        for key, value in classes_observations_dict.items():
            classes_col.append(key)
            observation_col.append(value)
        return pd.DataFrame.from_dict({output_class_col: classes_col, output_observation_col: observation_col})








