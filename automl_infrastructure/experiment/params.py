from abc import ABC, abstractmethod


class ParameterSuggestor(ABC):

    @abstractmethod
    def suggest_continous_float(self, name, low, high, log):
        pass

    @abstractmethod
    def suggest_discrete_float(self, name, low, high, step):
        pass

    @abstractmethod
    def suggest_int(self, name, low, high):
        pass

    @abstractmethod
    def suggest_list(self, name, options):
        pass


class OptunaParameterSuggestor(ParameterSuggestor):

    def __init__(self, trial):
        self._trial = trial

    def suggest_continous_float(self, name, low, high, log=False):
        if log:
            return self._trial.suggest_loguniform(name, low, high)
        else:
            return self._trial.suggest_uniform(name, low, high)

    def suggest_discrete_float(self, name, low, high, step):
        return self._trial.suggest_discrete_uniform(name, low, high, step)

    def suggest_int(self, name, low, high):
        return self._trial.suggest_int(name, low, high)

    def suggest_list(self, name, options):
        return self._trial.suggest_categorical(name, options)


class Parameter(ABC):

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def set_name(self, name):
        self._name = name

    @abstractmethod
    def suggest(self, suggestor):
        pass

    @abstractmethod
    def copy(self):
        pass


class RangedParameter(Parameter):

    def __init__(self, name, lower, upper, discrete=False, step_rate=None, log=False):
        super().__init__(name)
        self._lower = lower
        self._upper = upper
        self._discrete = discrete
        self._step_rate = step_rate
        self._log = log

    def suggest(self, suggestor):
        # check weather we deal with floats or ints
        if isinstance(self._lower, int) and isinstance(self._upper, int):
            return suggestor.suggest_int(self.name, self._lower, self._upper)
        else:
            effective_lower = float(self._lower)
            effective_upper = float(self._upper)
            # check weather discrete is needed or continous one
            if self._discrete:
                return suggestor.suggest_discrete_float(self.name, effective_lower, effective_upper, self._step_rate)
            else:
                return suggestor.suggest_continous_float(self.name, effective_lower, effective_upper, log=self._log)

    def copy(self):
        return RangedParameter(self._name, self._lower, self._upper, self._discrete, self._step_rate, self._log)


class ListParameter(Parameter):

    def __init__(self, name, options):
        super().__init__(name)
        self._options = options

    def suggest(self, suggestor):
        return suggestor.suggest_list(self.name, self._options)

    def copy(self):
        return ListParameter(self._name, self._options)


