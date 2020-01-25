# coding=UTF-8
"""heart of the cartesian genetic algorithm """

import os
import timeit
from abc import ABC, abstractmethod
from json import JSONEncoder
from fitness import NFitness

from solution import Solution


#pylint: disable=W0223,W0231
class Algorithm(ABC):
    """     abstract class for algorithm
            +----------------------+
            |                      |
            |   Initialization     |    > Creation of individuals
            |                      |
            +-----------+----------+
            |           |          |
            +-----------v----------+
            |                      |
            |  Fitness evaluation  |    > Fitness evaluation of the individuals
            |                      |
            +-----------+----------+
                        |
            +-----------v----------+
            |                      |
    +------->  Parent selection    |    > selection of the individuals for variation
    |       |                      |
    |       +-----------+----------+
    |                   |
    |       +-----------v----------+
    |       |                      |
    |       |  Variation           |    > Recombining
    |       |                      |    > Mutation
    |       +-----------+----------+
    |       |           |          |
    |       +-----------v----------+
    |       |                      |
    |       | Fitness evaluation   |
    |       |                      |
    |       +-----------+----------+
    |                   |
    |       +-----------v----------+
    |       |                      |
    +-------+  Survivor selection  |    > selection of the individuals for the next
            |                      |     generation (child selection)
            +-----------+----------+
                        |
            +-----------v----------+
            |                      |
            |  End                 |    > start of postprocessing
            |                      |
            +----------------------+

    :version:
    0.0.6-0
    :author:
    winter christoph

    """

    @abstractmethod
    def __init__(self, UserDecisions=None):
        self.time_algorithm = 0
        self.time_storage = 0
        self.epoch = 0
        self.generations = []
        self.user_decisions = UserDecisions
        self.randomstate = self.user_decisions.randomstate
        self.problem = UserDecisions.get_problem()

        ## reset used params from userconfig with used problem data
        self.user_decisions.reset_problem(self.problem)

        ## export json with right parameter
        self.user_decisions.export_json()

        self.procedure()


    @abstractmethod
    def init(self):
        """
        init solution based on algorithm
        Creation of individuals
        """

    @abstractmethod
    def parent_selection(self, generation):
        """
        selection of the individuals for variation, mu, l , mu + l or turnament
        mu (int):
        lambda (int): number of outputs of a problem
        elitism (bool):
            elitism: best individual is handed over to next gen unchanged
            n-elitism: the best n individuals are passed on to the next gen unchanged
        plus (int): mu + Lambda
        """

    @abstractmethod
    def survivor_selection(self, children):
        """
        selection of the individuals for the next
        generation (child selection)
        """

    @abstractmethod
    def variation(self, parents):
        """
        Recombining
        Mutation
        """

    def procedure(self):
        """ definition of how the problem is handeled as in the above graphic """
        start = timeit.default_timer()
        if self.user_decisions.verbose:
            print('>>\tinit')
        self.init()

        current_pop = self.generations[0]
        if self.user_decisions.verbose:
            print('>>\tloop selection variation evaluation selection')
        for _gen in range(self.user_decisions.max_generations):

            sel_par = self.parent_selection(current_pop)
            create_children = self.variation(sel_par)

            # ToDo: create a specific survivor selection
            survivors = self.parent_selection(create_children)

            self.generations.append(survivors)
            self.epoch = self.epoch + 1
            current_pop = survivors

        stop = timeit.default_timer()
        self.time_algorithm = stop - start
        if self.user_decisions.verbose:
            print('>>\tpostprocessing')
        start = timeit.default_timer()
        self.post_processing()
        stop = timeit.default_timer()

        self.time_storage = stop - start
        self.write_summary()

        # zip results
        if self.user_decisions.log.log_data('XXL'):
            self.user_decisions.log.zip_log()

        # print summery of parameters
        if self.user_decisions.verbose:
            print(str(self))


    @abstractmethod
    def post_processing(self):
        """
        use defined logging structure
        plot fitness statistics
        zip results
        """

    def timed_fitness(self, solution):
        """ return time for evaluation of one solution"""
        start = timeit.default_timer()
        stat2 = NFitness(operators=self.user_decisions.operators,
                         fitness_fct=self.user_decisions.fitness_fct)
        stat2.calc_fitness(solution, self.problem, self.user_decisions.evaluators)
        end = timeit.default_timer()

        print('Time for {0} evaluator {1:.4f}'.format(self.user_decisions.fitness_fct, end-start))
        for k, v in stat2.test_fitness.items():
            print("{0:>7}:\tTraining fitness {1:.4f}, \
                   Test fitness {2:.4f}".format(k, stat2.training_fitness[k], v))

    def write_summary(self):
        """ write summery of parameters """
        path = os.path.join(self.user_decisions.log.logfile_path, 'cgp_config' + '.txt')
        file = open(path, 'w')
        file.write(self.user_decisions.log.filename +'\n')
        file.write(str(self))
        file.close()


    def dump(self, name='default'):
        """ stores json of object """
        path = os.path.join(self.user_decisions.log.logfile_path, '{0}.json'.format(name))
        json_string = AlgorithmEncoder().encode(self)
        file = open(path, 'w')
        file.write(json_string)
        file.close()

    def check_rng(self):
        print("{0} : {1}".format(self.user_decisions.random_seed, self.randomstate.randint(10)))

class AlgorithmEncoder(JSONEncoder):
    """ parameters which the user may decide to use """
    # https://github.com/PyCQA/pylint/issues/414
    def default(self, object):  # pylint: disable=E0202
        if isinstance(object, (Algorithm)):
            return object.__dict__
        elif isinstance(object, (Solution)):
            return object.complete_solution
        else:
            return None
