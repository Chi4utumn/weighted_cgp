# coding=UTF-8
""" methods for fitness eval """

import os
import platform
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np

SCIPY = False
SKLEARN = False

try:
    from scipy.stats.stats import pearsonr
    from scipy.stats import chisquare
    SCIPY = True
except ImportError:
    print("scipy not available on {0}".format(platform.platform()))

try:
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    SKLEARN = True
except ImportError:
    print("sklearn not available on {0}".format(platform.platform()))


class Fitness(ABC):
    """ minimum fitness function needs to have """
    def __init__(self):
        self.fitness_fct = None
        self.training_fitness = None
        self.test_fitness = None

    def get_pearson_r(self, y_true, y_pred):
        """
        perfect fit result = 1 , with p CI of 0
        inverted fit result = -1 , with p CI of 0
        const leads to NaN , with p CI of 1
        doesn't change if calc and true values are mixed up
        no indicator of the scale
        """
        const_solution = all(x == y_pred[0] for x in y_pred)
        has_nan = any(np.isnan(y_pred))
        has_inf = any(np.isinf(y_pred))

        if const_solution or has_nan or has_inf:
            # return r = nan without calc and p = no corr = 1
            return np.nan, np.nan

        else:
            _r, _p = pearsonr(y_pred, y_true)
            return _r, _p

    def get_pearson_r2(self, y_true, y_pred):
        """
            test
        """
        const_solution = all(x == y_pred[0] for x in y_pred)

        if not const_solution:
            _r, _p = self.get_pearson_r(y_true=y_true, y_pred=y_pred)

            if _p > 0.15:
                return np.nan
            else:
                return _r**2
        else:
            # return nan without calc
            return np.nan

    def get_mean_squared(self, y_true, y_pred):
        """
        mean squared error:
            perfect fit result = 0, higher value is worse
            doesn't change if calc and true values are mixed up
        """
        try:
            return mean_squared_error(y_pred, y_true)
        except ZeroDivisionError:
            return  np.nan
        except Exception as _e:
            return  np.nan

    def get_mean_absolute(self, y_true, y_pred):
        """
        mean absolute error:
            perfect fit result = 0, higher value is worse
            doesn't change if calc and true values are mixed up
        """
        try:
            return mean_absolute_error(y_pred, y_true)
        except:
            return np.nan

    def get_r2_score(self, y_true, y_pred):
        """
        Args:
            y_true
            y_pred

        perfect fit result = 1, lower value is worse
        """
        try:
            return r2_score(y_true, y_pred)
        except:
            return np.nan

    def get_variance_accounted_for(self, y_true, y_pred):
        """
        Args:
            y_true, o, observed values
            y_pred, e, predicted values

        vaf(e, o) =  1 - var(o - e) / var(o)
        """
        try:
            return 1 - np.var(np.subtract(y_true, y_pred)) / np.var(y_true)
        except:
            return np.nan


    def get_chi_square(self, y_true, y_pred):
        """
        chisquared:
        Args:
            observed values frequencies
            expected values frequencies
            degrees of freedom, v = k - 1 (k len of each array)
        Returns
        X² Value
        p-value
        ref:
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
        - https://specminor.org/2017/01/08/performing-chi-squared-gof-python.html
        - https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/chi-square/
        - https://sciencing.com/info-8027315-degrees-freedom-chisquare-test.html

        - https://www.crashkurs-statistik.de/welchen-statistischen-test-soll-ich-waehlen/
        - https://www.crashkurs-statistik.de/
          chi-quadrat-test-abhaengigkeit-zwischen-zwei-nominalen-variablen-testen/
        - https://www.crashkurs-statistik.de/chi-quadrat-koeffizient-und-kontingenzkoeffizient-k/
        - https://rcompanion.org/rcompanion/b_03.html
        """
        try:
            # ToDo fix warning if value in y_true == 0
            return chisquare(y_pred, y_true)
        except:
            return np.nan

class NFitness(Fitness):
    """ instance of fitness N for new, but not needed as abstract after rework"""
    #ToDo decide if Fitness needs to be abstract
    def __init__(self, fitness_fct, operators, ):
        super()
        self.fitness_fct = fitness_fct
        self.operators = operators
        self.training_fitness = None
        self.test_fitness = None

    def calc_fitness(self, solution, problem, evaluators):
        """ fintess calc for train and test data """
        train_calc, train_true = self.get_data(solution, problem, data_type='training')
        test_calc, test_true = self.get_data(solution, problem, data_type='test')

        self.training_fitness = self.get_fitness(train_calc, train_true, evaluators) # train
        self.test_fitness = self.get_fitness(test_calc, test_true, evaluators) # test

    def get_data(self, solution, problem, data_type='training'):
        """ returns eval and true data """

        jobs = {'training' : problem.trainingdata,
                'test' : problem.testdata
               }

        nodes_for_output = list(solution.get_active())

        # sperate nodes to be calculated and input nodes
        nodes_for_calc = list(set(nodes_for_output) - set(solution.inputs.keys()))
        # remove output keys
        nodes_for_calc = list(set(nodes_for_calc) - set(solution.outputs.keys()))
        nodes_for_calc.sort()

        # list of inputs needed for calc
        inputs_variable = list(solution.inputs.keys())[len(solution.constants):]
        inputs_for_calc = list(set(nodes_for_output).intersection(set(inputs_variable)))
        inputs_for_calc.sort()

        problem_data = np.array(jobs[data_type])
        input_range = len(solution.inputs) - len(solution.constants)
        odata = problem_data[:, input_range:]
        idata = problem_data[:, :input_range]
        results = []
        solved_nodes = {}
        solved_nodes.update(solution.inputs)
        # init input values from problem for calculation
        # for every row in problem data

        # assign index as dict to active nodes
        used_for_sol = dict(zip(range(len(nodes_for_output)), nodes_for_output))
        # change value to key of dict
        array_idx = {v:k for k, v in used_for_sol.items()}

        # dict of genes that need to be calculated
        solve = {k:v for k, v in solution.solution.items() if k in nodes_for_calc}

        # ToDo: Check if inputs are really used
        # inputs used for solution size depends on jobs length
        evaluated = np.array([])
        for _e in list(set(nodes_for_output) &  set(solution.inputs.keys())):
            if _e in list(solution.constants.keys()):
                input_values = np.full(len(jobs[data_type]), solution.inputs[_e])
            else:
                input_values = idata[:, _e - len(solution.constants)]

            if evaluated.size == 0:
                evaluated = input_values
            else:
                evaluated = np.vstack((evaluated, input_values))

        for node in solve.values():
            _op = node[0]
            weights = node[1::2]

            idxs = [array_idx[i] for i in node[2::2]]

            eval_nodes = []
            for i, _e in enumerate(idxs):

                if len(evaluated.shape) == 1:
                    selected = evaluated
                else:
                    selected = evaluated[_e, :]
                eval_nodes.append(selected * weights[i])

            node_values = self.operators.use(_op, eval_nodes, bulk=True)

            # algorithm found no valid solution
            if all(np.isnan(x) for x in node_values):
                evaluated = np.empty_like(node_values)
                evaluated.fill(np.nan)
                evaluated = evaluated.reshape(1, len(node_values))
                break

            evaluated = np.vstack((evaluated, node_values))

        # if output points directly to an input array is out of shape
        if len(evaluated.shape) == 1:
            results = evaluated
        else:
            results = list(evaluated[-1, :])

        return results, odata.ravel()

    def get_fitness(self, training_result, true_result, evaluators):
        """ calculate fitness for given evaluators of training and test dataset"""
        fitnesses = {}

        if '95p' in evaluators:
            fitnesses['95p'] = np.percentile([abs(true_result[i][0] - e)
                                              for i, e in enumerate(training_result)], 95)
        if 'mean' in evaluators:
            fitnesses['mean'] = np.mean([abs(true_result[i][0] - e)
                                         for i, e in enumerate(training_result)])
        if 'stddev' in evaluators:
            fitnesses['stddev'] = np.std([abs(true_result[i][0] - e)
                                          for i, e in enumerate(training_result)])
        if 'variance' in evaluators:
            fitnesses['variance'] = np.var([abs(true_result[i][0] - e)
                                            for i, e in enumerate(training_result)])

        if 'pearson' in evaluators:
            fitnesses['pearson'],\
            fitnesses['pearson_p'] = self.get_pearson_r(y_true=true_result,
                                                        y_pred=training_result)
        if 'pearson_r2' in evaluators:
            fitnesses['pearson_r2'] = self.get_pearson_r2(y_true=true_result,
                                                          y_pred=training_result)
        if 'mae' in evaluators:
            fitnesses['mae'] = self.get_mean_absolute(y_true=true_result, y_pred=training_result)
        if 'mse' in evaluators:
            fitnesses['mse'] = self.get_mean_squared(y_true=true_result, y_pred=training_result)
        if 'r2_score' in evaluators:
            fitnesses['r2_score'] = self.get_r2_score(y_true=true_result, y_pred=training_result)
        if 'chisquare' in evaluators:
            fitnesses['chisquare'], fitnesses['chisquare_p'] = self.get_chi_square(y_true=true_result,
                                                                                   y_pred=training_result)
        if 'vaf' in evaluators:
            fitnesses['vaf'] = self.get_variance_accounted_for(y_true=true_result,
                                                               y_pred=training_result)

        return fitnesses

    def values_scatter(self, solution, problem, path):
        """ returns plot with true data vs calculated """
        fig = plt.figure(figsize=(19.2, 10.8))

        train_pred, train_true = self.get_data(solution, problem, 'training')
        test_pred, test_true = self.get_data(solution, problem, 'test')

        ref = np.hstack((test_true, train_true))
        ax1 = fig.add_subplot(111)

        ax1.scatter(test_true,
                    test_pred,
                    c='r', marker="s", label='test')
        ax1.scatter(train_true,
                    train_pred,
                    c='b', marker="+", label='training')
        ax1.scatter(ref,
                    ref,
                    linestyle='dashed', label='ref')

        plt.legend(loc='upper right',prop={'size':30},markerscale=6)
        plt.rc('legend', fontsize="x-large")

        plt.xlabel('f(x)_expected')
        plt.ylabel('f(x)_calculated')
        plt.title('fitness of ' + problem.name)
        plt.grid()

        path = os.path.join(path, 'data_true_vs_pred.png')
        fig.savefig(path)
