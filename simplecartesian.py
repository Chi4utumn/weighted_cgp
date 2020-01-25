"""
cgp with weights uses mu + Lambda
    - optional constants
    - several fitness evaluators
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from algorithm import Algorithm
from candidate import Parent
from mutate import PointMutate
from fitness import NFitness

class SimpleCartesian(Algorithm):
    """ implementation of cartesian"""
    def __init__(self, UserDecisions):
        Algorithm.__init__(self, UserDecisions=UserDecisions)

    def init(self):
        """ Initialization: init of solutions for pop size """
        # keep matrix the same for simple crossover
        rows = self.user_decisions.rows
        columns = self.user_decisions.columns
        self.shape = '{0}x{1}'.format(rows, columns)

        generation_zero = []
        for _x in range(self.user_decisions.population_size):
            current_solution = Parent(params=self, rows=rows, columns=columns)

            generation_zero.append(current_solution)

        self.generations.append(generation_zero)

    def parent_selection(self, generation):
        """ use 1 + 4 ES"""
        # sort by fitness
        # Todo fitness selector currently only 'best'
        # min values are better
        if self.user_decisions.fitness_fct in ['mae', 'mse']:
            survivors_0 = sorted(generation,
                                 key=lambda x: x.training_fitness[self.user_decisions.fitness_fct],
                                 reverse=False)

        # max values are better
        if self.user_decisions.fitness_fct in ['r2_score', 'vaf']:
            filter_nan = [x for x in generation
                          if not np.isnan(x.training_fitness[self.user_decisions.fitness_fct])]
            survivors_0 = sorted(filter_nan,
                                 key=lambda e: e.training_fitness[self.user_decisions.fitness_fct],
                                 reverse=True)

        # other values are better...
        if self.user_decisions.fitness_fct in ['pearson']:
            # 1st CI if smaller 0.05: it is significant
            # 2nd value sorted should be far from zero, negatives represent negative corr
            alpha = 0.1
            if self.epoch == 0:
                alpha = 0.9
            remove_not_in_ci = [x for x in generation if x.training_fitness['pearson_p'] <= alpha]
            survivors_0 = sorted(remove_not_in_ci,
                                 key=lambda e: e.training_fitness[self.user_decisions.fitness_fct],
                                 reverse=True)

            if len(survivors_0) < 1:
                sort_r = sorted(generation,
                                key=lambda e: e.training_fitness[self.user_decisions.fitness_fct],
                                reverse=True)
                survivors_0 = sorted(sort_r,
                                     key=lambda e: (e.training_fitness['pearson_p'],
                                                    e.training_fitness['mae']))

        if self.user_decisions.fitness_fct in ['chisquare']:
            remove_negative = [x for x in generation
                               if x.training_fitness[self.user_decisions.fitness_fct] >= 0]
            chi_xx = sorted(remove_negative,
                            key=lambda e: (e.training_fitness['chisquare_p']),
                            reverse=True)
            survivors_0 = sorted(chi_xx,
                                 key=lambda e: (e.training_fitness[self.user_decisions.fitness_fct]))

        # squared pearson for simpler selection of fittest
        if self.user_decisions.fitness_fct in ['pearson_r2']:
            survivors_0 = sorted(generation,
                                 key=lambda e: (e.training_fitness[self.user_decisions.fitness_fct]),
                                 reverse=True)


        mu_es = self.user_decisions.es_mu
        lambda_es = self.user_decisions.es_lambda

        # first mu +lambda solution are chosen
        result = survivors_0[:mu_es+lambda_es]

        # If an offspring genotype has a better or equal fitness than the parent then
        # Offspring genotype is chosen as fittest

        # rank to bottom
        if self.epoch > 0:
            #ToDo clean up, only needed if parent has same fitness as others
            children = [x.training_fitness for x in result[1:]]
            parent_fitness = result[0].training_fitness
            samefitness_pos = [i+1 for i, x in enumerate(children) if x == parent_fitness]

            if samefitness_pos:
                result.insert(samefitness_pos[-1], result.pop(0))
                # sort again by fitness
                # result = sorted(result, key=lambda x: x.training_fitness, reverse=False)

        return result

    def survivor_selection(self, children):
        """ use 1 + 4 ES"""
        raise NotImplementedError

    def variation(self, parents):
        """apply variation """
        children = []

        # mu parents (1) are chosen for reproduction
        children.append(parents[0])

        # lambda children (4) are created from parent
        for _j in range(self.user_decisions.es_lambda):

            new_ind = PointMutate(parents[0], self.randomstate)
            # self.write_json(str(new_ind.new_child.id),new_ind.new_child)
            new_ind.new_child.complete_the_solution()
            children.append(new_ind.new_child)

        return children

    def post_processing(self):

        # store dot and png

        # ToDo dependent on evaluator r2_score,vaf best is 1 mse,mae best is 0

        best = self.generations[-1][0]
        best.store_dot_file(name='best')

        # Todo add operators
        best.dump('best')

        best_fitness = NFitness(self.user_decisions.fitness_fct, self.user_decisions.operators)
        best_fitness.values_scatter(best, self.problem, self.user_decisions.log.logfile_path)
        best_train_eval, _best_train_true = best_fitness.get_data(best, self.problem, data_type='training')
        best_test_eval, _best_test_true = best_fitness.get_data(best, self.problem, data_type='test')

        # ToDo add if for verbosity level
        if self.user_decisions.verbose:
            print()
            print('Fit train:\t{0} \t test: {1} '.format(best.training_fitness,
                                                         best.test_fitness))
            print()

            self.timed_fitness(self.generations[-1][0])
            print()

        if self.user_decisions.log.log_data('M'):

            self.fitness_per_gen_plot()

        if self.user_decisions.log.log_data('L'):
            self.problem.path = self.user_decisions.log.logfile_path
            self.problem.get_problem_df(best_train_eval, best_test_eval)

            # self.problem.to_csv(
            #    best_train_eval,
            #    best_test_eval,
            #    self.user_decisions.log.logfile_path)

        if self.user_decisions.log.log_data('XXL'):
            self.problem.scatter()

        if self.user_decisions.log.log_data('XL'):

            self.problem.multiscatter(
                best_train_eval,
                best_test_eval,
                self.user_decisions.log.logfile_path)

        # caution dumps all solution dicts in huge json file
        #items still missing...
        if self.user_decisions.log.log_data('XXXL'):
            self.dump()

    def fitness_per_gen_plot(self):
        fittest_per_gen = []
        fitness_all_gens = []

        fittest_dict = {}
        count = 0
        corresponding_test_fit = {}

        for _i, gen in enumerate(self.generations):

            # sorted already in parent selection therefore fittest is always min of list
            min_per_gen = min(gen, key=lambda x: x.training_fitness[self.user_decisions.fitness_fct])

            idx_min = gen.index(min_per_gen)

            fittest_dict[count + idx_min] = min_per_gen.training_fitness[self.user_decisions.fitness_fct]
            corresponding_test_fit[count + idx_min] = min_per_gen.test_fitness[self.user_decisions.fitness_fct]

            for j, _s in enumerate(gen):
                if j == 0:
                    fittest_per_gen.append(_s.training_fitness[self.user_decisions.fitness_fct])

                fitness_all_gens.append(_s.training_fitness[self.user_decisions.fitness_fct])

            count = count + len(gen)

        # remove outliers in  range mean +- 1 std dev
        fitness_stddev = np.std(fitness_all_gens)
        fitness_mean = np.mean(fitness_all_gens)

        xmin = fitness_mean - 1 * fitness_stddev
        xmax = fitness_mean + 1 * fitness_stddev

        dict_all_gen_fit = {i:x for i, x in enumerate(fitness_all_gens) if (x <= xmax and x >= xmin)}
        dict_all_gen_fit_outlier = {i:0 for i, x in enumerate(fitness_all_gens) if (x > xmax or x < xmin)}


        fig = plt.figure(figsize=(19.2, 10.8))
        ax1 = fig.add_subplot(111)
        ax1.scatter(list(dict_all_gen_fit_outlier.keys()),
                    list(dict_all_gen_fit_outlier.values()),
                    c='r', marker="|", label='outlier')
        ax1.scatter(list(dict_all_gen_fit.keys()),
                    list(dict_all_gen_fit.values()),
                    c='b', marker="+", label='training')
        ax1.scatter(list(fittest_dict.keys()),
                    list(fittest_dict.values()),
                    c='orange', marker="p", label='fittest/gen')
        ax1.scatter(list(corresponding_test_fit.keys()),
                    list(corresponding_test_fit.values()),
                    c='g', marker="x", label='testdata')

        plt.legend(loc='upper right',prop={'size':30},markerscale=6)
        plt.rc('legend',fontsize="x-large")
        plt.xlabel('solution number (#)')
        plt.ylabel(self.user_decisions.fitness_fct + ' fitness value')
        plt.title('fitness of ' + self.problem.name)
        plt.grid()

        path = os.path.join(self.user_decisions.log.logfile_path, 'fitness_over_gen.png')
        fig.savefig(path)

        if self.user_decisions.show_plt:
            plt.show()

    def fitness_per_gen_csv(self, parameter_list):
        pass

    def __str__(self):
        """ override self representation: """
        # ToDo format section
        return str(self.__class__.__name__) + ':\n' +       \
                'Problem Name:\t\t' + str(self.problem.name) + '\n' + \
                'Sol shape:\t\t{0}\n'.format(self.shape) + \
                '# of Sols:\t\t' + str(sum([len(x) for x in self.generations])) + '\n' + \
                '# of Gens:\t\t' + str(len(self.generations)) + '\n' + \
                'time needed algorithm:\t{0}'.format(self.time_algorithm) + '\n' + \
                'time needed post algorithm:\t{0}'.format(self.time_storage) + '\n' + \
                '\n' + str(self.user_decisions)  + '\n' + \
                'fittest:\t ' + str(self.generations[-1][0])
