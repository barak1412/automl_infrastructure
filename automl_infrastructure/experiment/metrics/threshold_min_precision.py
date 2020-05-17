from automl_infrastructure.experiment.metrics import SimpleMetric
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import LabelBinarizer
from collections import OrderedDict
import numpy as np

class ThresholdMinPrecision(SimpleMetric):

    def __init__(self, precision, is_grouped=False, weighted=True):
        super().__init__(is_grouped=is_grouped, weighted=weighted)
        self._precision = precision

    def measure_lst(self, y_true, classifier_prediction):
        # transform classes to binary vectors
        classes = [c for c in y_true] + [c for c in classifier_prediction.classes_pred]
        classes = sorted(list(set(classes)))
        binarizer = LabelBinarizer()
        binarizer.fit(classes)
        y_true_binary = binarizer.transform(y_true)

        # generate metric for every class
        classes_value = OrderedDict()
        if(len(classes)==2):
            y_true_binary = np.column_stack([(1- y_true_binary)[:,0],y_true_binary[:,0]])
        
        for idx, c in enumerate(classes):
            precision_lst, recall_lst, ticks = precision_recall_curve(y_true_binary[:, idx],
                                                                      classifier_prediction.classes_proba[:, idx])
            classes_value[c] = ThresholdMinPrecision._find_threshold_above(ticks, recall_lst, precision_lst, self._precision)
        return classes_value

    @staticmethod
    def _find_threshold_above(ticks, indicators, values, threshold_value):
        value = 0.0
        best_threshold = 1.01
        for i in range(len(ticks)):
            if threshold_value <= values[i] and indicators[i] > value:
                value = indicators[i]
                best_threshold = ticks[i]
        return best_threshold

