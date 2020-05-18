import copy
import operator
from functools import reduce
from time import strftime, gmtime

import dill
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from automl_infrastructure.classifiers import ClassifierPrediction
from automl_infrastructure.experiment.metrics.standard_metrics import ObjectiveFactory
from automl_infrastructure.experiment.params import OptunaParameterSuggestor
from automl_infrastructure.utils import random_str


class Experiment(object):

    def __init__(self, name, x, y, models, hyper_parameters={}, observations={}, visualizations={},
                 objective=None, objective_name=None, maximize=True,
                 n_folds=3, n_repetitions=5, additional_training_data_x=None, additional_training_data_y=None):
        self._name = name
        self._x = x
        self._y = y
        self._additional_training_data_x = additional_training_data_x
        self._additional_training_data_y = additional_training_data_y
        self._models = models
        self._hyper_parameters = hyper_parameters
        self._observations = observations
        self._visualizations = visualizations
        self._objective_name = objective_name
        self._objective = self._parse_objective(objective)
        self._objective_direction = 'maximize' if maximize else 'minimize'

        # set parameters for repeated k-fold cross validation
        self._n_folds = n_folds
        self._n_repetitions = n_repetitions

        # initialize output results
        self._models_test_observations = {}
        self._models_train_observations = {}
        self._models_test_visualizations = {}
        self._models_train_visualizations = {}
        self._models_best_params = {}
        self._models_best_scores = {}
        self._models_train_best_scores = {}
        self._best_model_name = None

        self._start_time = None
        self._end_time = None

    @property
    def X(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def additional_training_data_X(self):
        return self._additional_training_data_x

    @property
    def additional_training_data_y(self):
        return self._additional_training_data_y

    @property
    def objective_name(self):
        return self._objective_name

    @property
    def best_model(self):
        best_model = None
        for model in self._models:
            if model.name == self._best_model_name:
                best_model = model
        best_model.set_params(self._models_best_params[best_model.name])
        return best_model

    def objective_score(self, model_name, group='test'):
        if group == 'train':
            best_scores = self._models_train_best_scores
        elif group == 'test':
            best_scores = self._models_best_scores
        else:
            raise Exception('Unsupported group {}, only [\'train\', \'test\'] are supported.'.format(group))

        if model_name not in best_scores:
            raise ('Could not find model named {}.'.format(model_name))
        return best_scores[model_name]

    @property
    def end_time(self):
        return self._end_time

    def _parse_objective(self, objective):
        if callable(objective):
            if self._objective_name is None:
                raise Exception('You must specified objective_name for callable objective function.')
            return objective
        elif isinstance(objective, str):
            if self._objective_name is None:
                self._objective_name = objective
            objective = ObjectiveFactory.create(objective)
            return lambda y_true, classifier_prediction: objective.measure(y_true, classifier_prediction)
        else:
            raise Exception('Only objective name or callable are supported.')

    @staticmethod
    def _build_hyper_params_translation(model_name, hyper_params):
        model_hyper_params = hyper_params[model_name]
        translation_result = {}
        new_params_flat = {}

        # check leaf
        if not isinstance(model_hyper_params, dict):
            new_params_hirerachy = []
            for param in model_hyper_params:
                random_name = '{0}_{1}'.format(param.name, random_str(length=5))
                translation_result[random_name] = param.name
                new_param = param.copy()
                new_param.set_name(random_name)
                new_params_flat[random_name] = new_param
                new_params_hirerachy.append(new_param)
        else:
            new_params_hirerachy = {}
            for key in model_hyper_params.keys():
                translation_iter, new_params_flat_iter, new_params_hirerachy[
                    key] = Experiment._build_hyper_params_translation(key, model_hyper_params)
                translation_result.update(translation_iter)
                new_params_flat.update(new_params_flat_iter)

        return translation_result, new_params_flat, new_params_hirerachy

    @staticmethod
    def _translate_best_params(new_hyper_parameters, translation, best_params):
        result = {}
        if not isinstance(new_hyper_parameters, dict):
            for param, value in best_params.items():
                new_hyper_parameters_names = [p.name for p in new_hyper_parameters]
                if param in new_hyper_parameters_names:
                    result[translation[param]] = value
        else:
            for model_name, inner_hyper_parameters_dict in new_hyper_parameters.items():
                result[model_name] = Experiment._translate_best_params(inner_hyper_parameters_dict, translation,
                                                                       best_params)

        return result

    def run(self, n_trials=3, n_jobs=15):
        # init starting time
        self._start_time = gmtime()
        for model in self._models:
            # check weather model have hyper-parameters
            if model.name not in self._hyper_parameters or n_trials <= 0:
                best_params = {}
            else:
                # translate params names
                translation, new_params_flat, new_params_hierarchy = Experiment._build_hyper_params_translation(
                    model.name,
                    self._hyper_parameters)
                best_params = self._optimize_model(model, n_jobs, self._n_folds, self._n_repetitions, n_trials,
                                                   new_params_flat, new_params_hierarchy, translation)
                best_params = Experiment._translate_best_params(new_params_hierarchy, translation, best_params)
            self._models_best_params[model.name] = best_params

        # update observations and best scores
        self.refresh()

        # init end time
        self._end_time = gmtime()

    def _optimize_model(self, model, n_jobs, n_folds, n_repeatations, n_trials, new_params_flat, new_params_hierarchy,
                        translation):

        def optuna_objective(trial):
            # set params of model
            optuna_suggestor = OptunaParameterSuggestor(trial)
            suggested_params = {}
            for new_param_name, new_param in new_params_flat.items():
                suggested_params[new_param_name] = new_param.suggest(optuna_suggestor)
            suggested_params_translated = Experiment._translate_best_params(new_params_hierarchy, translation,
                                                                            suggested_params)

            # copy model for multi-threaded support
            copied_model = copy.deepcopy(model)
            copied_model.set_params(suggested_params_translated)

            # repeat k fold
            for i in range(0, n_repeatations):
                # split training data to k folds
                k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True)
                scores = []
                for train_index, test_index in k_fold.split(self._x, self._y):
                    X_train, X_test = self._x.iloc[train_index], self._x.iloc[test_index]
                    y_train, y_test = self._y.iloc[train_index], self._y.iloc[test_index]
                    if self._additional_training_data_x is not None:
                        X_train = X_train.append(self._additional_training_data_x)
                        y_train = y_train.append(self._additional_training_data_y)
                        X_train, y_train = shuffle(X_train, y_train)
                    copied_model.fit(X_train, y_train.values.ravel())

                    # wrap prediction
                    pred_y = copied_model.predict(X_test)
                    proba_y = copied_model.predict_proba(X_test)
                    classifier_prediction = ClassifierPrediction(pred_y, proba_y)

                    score = self._objective(y_test.values.ravel(), classifier_prediction)
                    scores.append(score)
            return np.mean(scores)

        study = optuna.create_study(study_name='{0}_{1}'.format(self._name, model.name),
                                    direction=self._objective_direction)
        study.optimize(optuna_objective, n_trials=n_trials, n_jobs=n_jobs)
        return study.best_params

    def refresh(self):
        current_best_score = None
        for model in self._models:
            best_params = self._models_best_params[model.name]
            # set models params for observation creation
            model.set_params(best_params)

            # generate observations and score of model
            self._generate_model_results(model, self._n_folds, self._n_repetitions)

            # update best model
            best_score = self._models_best_scores[model.name]
            if current_best_score is None or (
                    self._objective_direction == 'maximize' and best_score > current_best_score) \
                    or (self._objective_direction == 'minimize' and best_score < current_best_score):
                self._best_model_name = model.name
                current_best_score = best_score

    def _generate_model_results(self, model, n_folds, n_repetitions):
        test_y_true_lst = []
        test_classifier_predictions_lst = []
        train_y_true_lst = []
        train_classifier_predictions_lst = []
        test_scores = []
        train_scores = []
        # repeat k fold
        for i in range(0, n_repetitions):
            # split training data to k folds
            k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True)
            for train_index, test_index in k_fold.split(self._x, self._y):
                # X_train, X_test = self._x[train_index, :], self._x[test_index, :]
                X_train, X_test = self._x.iloc[train_index], self._x.iloc[test_index]
                y_train, y_test = self._y.iloc[train_index], self._y.iloc[test_index]
                if self._additional_training_data_x is not None:
                    X_train = X_train.append(self._additional_training_data_x)
                    y_train = y_train.append(self._additional_training_data_y)
                    X_train, y_train = shuffle(X_train, y_train)
                model.fit(X_train, y_train.values.ravel())

                # wrap prediction for test
                pred_y = model.predict(X_test)
                proba_y = model.predict_proba(X_test)
                test_classifier_predictions = ClassifierPrediction(pred_y, proba_y)
                test_classifier_predictions_lst.append(test_classifier_predictions)
                test_y_true_lst.append(y_test.values.ravel())
                score = self._objective(y_test.values.ravel(), test_classifier_predictions)
                test_scores.append(score)

                # wrap prediction for train
                pred_y = model.predict(X_train)
                proba_y = model.predict_proba(X_train)
                train_classifier_predictions = ClassifierPrediction(pred_y, proba_y)
                train_classifier_predictions_lst.append(train_classifier_predictions)
                train_y_true_lst.append(y_train.values.ravel())
                score = self._objective(y_train.values.ravel(), train_classifier_predictions)
                train_scores.append(score)

        # set best score
        self._models_best_scores[model.name] = np.mean(test_scores)
        self._models_train_best_scores[model.name] = np.mean(train_scores)
        # set observations
        self._generate_model_observations(model.name, train_y_true_lst, train_classifier_predictions_lst,
                                          test_y_true_lst, test_classifier_predictions_lst)
        # set visualizations
        self._generate_model_visualizations(model.name, train_y_true_lst, train_classifier_predictions_lst,
                                            test_y_true_lst, test_classifier_predictions_lst)

    def _generate_model_observations(self, model_name, train_y_true_lst, train_classifier_predictions_lst,
                                     test_y_true_lst, test_classifier_predictions_lst):
        # generate observations and merge to one dataframe
        test_observations_dataframes = []
        train_observations_dataframes = []
        for observation in self._observations:
            observation_obj = self._observations[observation]

            # add observation to test
            df = observation_obj.observe(test_y_true_lst, test_classifier_predictions_lst, output_class_col='CLASS',
                                         output_observation_col=observation)
            test_observations_dataframes.append(df)

            # add observation to test
            df = observation_obj.observe(train_y_true_lst, train_classifier_predictions_lst, output_class_col='CLASS',
                                         output_observation_col=observation)
            train_observations_dataframes.append(df)

        self._models_test_observations[model_name] = \
            reduce(lambda x, y: pd.merge(x, y, on='CLASS'), test_observations_dataframes)
        self._models_train_observations[model_name] = \
            reduce(lambda x, y: pd.merge(x, y, on='CLASS'), train_observations_dataframes)

    def _generate_model_visualizations(self, model_name, train_y_true_lst, train_classifier_predictions_lst,
                                       test_y_true_lst, test_classifier_predictions_lst):
        self._models_train_visualizations[model_name] = {}
        self._models_test_visualizations[model_name] = {}
        for name, visualization in self._visualizations.items():
            # train
            train_visualization = copy.deepcopy(visualization)
            train_visualization.fit(train_y_true_lst, train_classifier_predictions_lst)
            self._models_train_visualizations[model_name][name] = train_visualization

            # test
            test_visualization = copy.deepcopy(visualization)
            test_visualization.fit(test_y_true_lst, test_classifier_predictions_lst)
            self._models_test_visualizations[model_name][name] = test_visualization

    def get_model_observations(self, model_name, observation_type='test'):
        if observation_type == 'test':
            return self._models_test_observations[model_name]
        elif observation_type == 'train':
            return self._models_train_observations[model_name]
        else:
            raise Exception('Unsupported objective_type {}, only train or test supported.'.format(observation_type))

    def get_model_visualizations(self, model_name, observation_type='test'):
        if observation_type == 'test':
            return self._models_test_visualizations[model_name]
        elif observation_type == 'train':
            return self._models_train_visualizations[model_name]
        else:
            raise Exception('Unsupported observation_type {}, only train or test supported.'.format(observation_type))

    def add_observation(self, name, observation):
        if name in self._observations:
            raise Exception('Unable to add observation: observation named {} already exist.'.format(name))
        self._observations[name] = observation

    def add_visualization(self, name, visualization):
        if name in self._visualizations:
            raise Exception('Unable to add visualization: visualization named {} already exist.'.format(name))
        self._visualizations[name] = visualization

    def remove_visualization(self, name):
        if name not in self._visualizations:
            raise Exception('Unable to remove visualization: visualization named {} does not exist.'.format(name))
        del self._visualizations[name]

    def print_report(self, print_func=print):
        print("Experiment's name: {}.".format(self._name))
        time_format = '%H:%M:%S - %d/%m/%y'
        print('Start time: {}.'.format(strftime(time_format, self._start_time)))
        print('End time: {}.'.format(strftime(time_format, self._end_time)))
        print('Num of folds: {}.'.format(self._n_folds))
        print('Num of k-folds repetitions: {}.'.format(self._n_repetitions))
        print()
        # sort models by score
        reverse = True if self._objective_direction == 'maximize' else False
        sorted_models_scores = [m[0] for m in
                                sorted(self._models_best_scores.items(), key=operator.itemgetter(1), reverse=reverse)]

        # print models results
        for model_name in sorted_models_scores:
            print('---------------------------------------------------------')
            print('Model name: {}.'.format(model_name))
            print('Score: {}.'.format(self._models_best_scores[model_name]))
            print()
            print('Best hyper-parameters: {}.'.format(self._models_best_params[model_name]))
            print()
            print("Train's observations:")
            print_func(self._models_train_observations[model_name])
            print()
            if len(self._visualizations) > 0:
                print("Train's visualizations:")
                for visualization in self._visualizations:
                    print('{}:'.format(visualization))
                    self._models_train_visualizations[model_name][visualization].show()
                print()
            print("Test's observations:")
            print_func(self._models_test_observations[model_name])
            if len(self._visualizations) > 0:
                print("Test's visualizations:")
                for visualization in self._visualizations:
                    print('{}:'.format(visualization))
                    self._models_test_visualizations[model_name][visualization].show()
                print()

            print()
        print('---------------------------------------------------------')

    def dump(self, path, add_date=True):
        effective_file_name = self._name
        if add_date:
            current_date = strftime('%d%m%y_%H%M%S', gmtime())
            effective_file_name = effective_file_name + '_{0}'.format(current_date)
        file_path = '{0}/{1}.pckl'.format(path, effective_file_name)
        with open(file_path, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            obj = dill.load(file)
            return obj
        return None
