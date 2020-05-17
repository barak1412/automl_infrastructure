from automl_infrastructure.visualization import Visualization
from automl_infrastructure.utils import extract_ordered_classes
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import numpy as np


class PrecisionRecallCurve(Visualization):

    def __init__(self, max_classes_per_figure=5, n_thresholds=None):
        self._classes_curves = {}
        self._max_classes_per_figure = max_classes_per_figure
        self._n_thresholds = n_thresholds

    @staticmethod
    def _custom_precision_recall_curve(y_true, probas_pred, threshold_start=0.0, threshold_finish=0.999,
                                  threshold_amount=50):
        x = np.linspace(threshold_start, threshold_finish, num=threshold_amount)
        precisions = []
        recalls = []
        for thres in x:
            current_pred = np.array(np.array(probas_pred) > thres, dtype=int)
            precision = precision_score(y_pred=current_pred, y_true=y_true)
            recall = recall_score(y_pred=current_pred, y_true=y_true)
            precisions.append(precision)
            recalls.append(recall)
        return precisions, recalls, x

    def fit(self, y_true_lst, classifier_prediction_lst):
        classes = extract_ordered_classes(y_true_lst, classifier_prediction_lst)
        y_true_combined = np.concatenate(y_true_lst)
        classes_proba_combined = np.concatenate([c.classes_proba for c in classifier_prediction_lst])
        binarizer = LabelBinarizer()
        binarizer.fit(classes)
        y_true_combined_binary = binarizer.transform(y_true_combined)

        
        if(len(classes)==2):
            y_true_combined_binary = np.column_stack([(1- y_true_combined_binary)[:,0],y_true_combined_binary[:,0]])
        # generate metric for every class
        for idx, c in enumerate(classes):
            if self._n_thresholds is None:
                precision_lst, recall_lst, ticks = precision_recall_curve(y_true_combined_binary[:, idx],
                                                                          classes_proba_combined[:, idx])
            else:
                precision_lst, recall_lst, ticks = PrecisionRecallCurve._custom_precision_recall_curve(y_true_combined_binary[:, idx],
                                                    classes_proba_combined[:, idx], threshold_amount=self._n_thresholds)
            self._classes_curves[c] = (precision_lst, recall_lst, ticks)

    def get_curve(self, class_name):
        return self._classes_curves[class_name]

    @property
    def classes_(self):
        return [k for k in self._classes_curves]

    def show(self):
        n = self._max_classes_per_figure
        for lim in range(0, len(self._classes_curves), n):
            i = 0
            plt.clf()
            plt.figure(figsize=(n, n))
            for c in self._classes_curves:
                if lim <= i < lim + n:
                    precision_lst, recall_lst, _ = self._classes_curves[c]
                    plt.plot(recall_lst, precision_lst, label=get_display(c))
                i += 1
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title('Precision-Recall Curve')
            plt.legend(loc='upper right')
            plt.pause(0.5)
