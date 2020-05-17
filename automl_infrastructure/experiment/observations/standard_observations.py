from automl_infrastructure.experiment.observations import SimpleObservation
import numpy as np


class Std(SimpleObservation):
    def __init__(self, metric):
        super().__init__(metric)

    def agg_func(self, values):
        return np.std(values)


class Avg(SimpleObservation):
    def __init__(self, metric):
        super().__init__(metric)

    def agg_func(self, values):
        return np.mean(values)

