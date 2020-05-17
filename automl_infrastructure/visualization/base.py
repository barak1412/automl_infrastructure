from abc import ABC, abstractmethod


class Visualization(ABC):

    @abstractmethod
    def fit(self, y_true_lst, classifier_prediction_lst):
        pass

    @abstractmethod
    def show(self):
        pass

