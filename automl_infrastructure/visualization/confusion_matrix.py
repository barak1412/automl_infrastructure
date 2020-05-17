from automl_infrastructure.visualization import Visualization
from automl_infrastructure.utils import extract_ordered_classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from bidi.algorithm import get_display


class ConfusionMatrix(Visualization):

    def __init__(self, sample_weight=None, order=None, normalize=True, figsize=None, custom_thresholds=None, other_class='other'):
        self._sample_weight = sample_weight
        self._order = order
        self._normalize = normalize
        self._confusion_matrix = None
        self._figsize = figsize
        self._custom_thresholds = custom_thresholds
        self._other_class = other_class
        self._confusion_columns = None

    def to_dict(self):
        d = {}
        for i in range(len(self._confusion_columns)):
            d[self._confusion_columns[i]] = {}
            for j in range(len(self._confusion_columns)):
                d[self._confusion_columns[i]][self._confusion_columns[j]] = self._confusion_matrix[i][j]
        return d

    def fit(self, y_true_lst, classifier_prediction_lst):
        # init classes order
        if self._order is None:
            self._order = extract_ordered_classes(y_true_lst, classifier_prediction_lst)
        self._confusion_columns = self._order.copy()
        if self._custom_thresholds is not None:
            self._confusion_columns.append(self._other_class)

        n = len(y_true_lst)
        accumulated_matrix = None

        # calculate average confusion matrix
        for i in range(n):
            if self._custom_thresholds is not None:
                classes_pred = self.repredict_other_label(classifier_prediction_lst[i].classes_proba, sorted(self._order))
            else:
                classes_pred = classifier_prediction_lst[i].classes_pred
            current_matrix = confusion_matrix(y_true_lst[i], classes_pred,
                                              labels=self._confusion_columns)
            if i == 0:
                accumulated_matrix = current_matrix
            else:
                accumulated_matrix += current_matrix
        accumulated_matrix = accumulated_matrix / n

        # normalize rows if needed
        if self._normalize:
            accumulated_matrix = normalize(accumulated_matrix, axis=1, norm='l1')
        self._confusion_matrix = accumulated_matrix

    def repredict_other_label(self, probs_lst, classes):
        y_pred = [self._other_class if l[np.argmax(l)] < self._custom_thresholds[classes[np.argmax(l)]]\
                      else classes[np.argmax(l)] for l in probs_lst]
        return y_pred

    @staticmethod
    def _create_matrix_figure(matrix_df, figsize):
        fig = plt.figure(figsize=figsize)
        plt.clf()
        plt.grid(b=None)
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(matrix_df.values, cmap=plt.cm.jet, interpolation='nearest', vmax=1.0, vmin=0.0, aspect='equal')
        width, height = matrix_df.shape
        for x in range(width):
            for y in range(height):
                ax.annotate(float('%.2f' % matrix_df.iloc[x][y]), xy=(y, x), \
                            horizontalalignment='center', verticalalignment='center')
        cb = fig.colorbar(res)
        #plt.xticks(range(width), matrix_df.columns.tolist())
        #plt.yticks(range(height), matrix_df.columns.tolist())
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.set_xticklabels(matrix_df.columns.tolist())
        ax.set_yticklabels(matrix_df.columns.tolist())
        #ax.grid(which='minor', color='w', linestyle='-')
        ax.xaxis.tick_top()
        return fig

    def show(self):
        if self._confusion_matrix is None:
            raise Exception('Can not plot unfitted visualization.')
        # set figsize
        if self._figsize is None:
            figsize = (len(self._order), len(self._order))
        else:
            figsize = self._figsize
        columns = [get_display(c) for c in self._confusion_columns]
        confusion_df = pd.DataFrame(self._confusion_matrix, index=columns, columns=columns)
        fig = ConfusionMatrix._create_matrix_figure(confusion_df, figsize)
        fig.show()
        plt.pause(0.5)

