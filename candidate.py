""" solution class instances
    - parents for init pop
    - child to create children from parents
"""
from copy import deepcopy
from solution import Solution
from fitness import NFitness

#pylint: disable=C0123
class Candidate(Solution):
    """
    Args:
        Empty Solution
    Returns:
        nothing by default
    Raises:
        no error handling
    """
    def __init__(self):
        """ ToDo write info text"""
        Solution.__init__(self)

class Child(Solution):
    """
    Args:
        Incomplete copy of a parent
    Returns:
        nothing by default
    Raises:
        no error handling

    """
    def __init__(self, parent):
        Solution.__init__(self)
        self.params = parent.params

        self.rows = deepcopy(parent.rows)
        self.columns = deepcopy(parent.columns)
        self.shape = self.get_shape()

        # minimum is one
        self.levels_back = deepcopy(parent.levels_back)

        # inputs
        # constants [] + inputs []
        self.constants = deepcopy(parent.constants)
        self.inputs = deepcopy(parent.inputs)

        # solution
        # sequence [] + output []
        self.outputs = deepcopy(parent.outputs)
        self.sequences = deepcopy(parent.sequences)

        self.solution = deepcopy(parent.solution)

    def complete_the_solution(self):
        """
        Child item copies parent, then gets mutated and therefore cant be computed on init
        sequences needed because only solution is mutated and get active uses sequences
        """
        self.sequences = self.sol2seq()
        self.active_nodes = self.get_active()
        statistics = NFitness(self.params.user_decisions.fitness_fct,
                              self.params.user_decisions.operators)

        statistics.calc_fitness(self, self.params.problem,
                                      self.params.user_decisions.evaluators)

        self.training_fitness = statistics.training_fitness
        self.test_fitness = statistics.test_fitness

        self.complete_solution = self.get_all_nodes()

    def sol2seq(self):
        """ reverse mutated solution to sequence
            essentially all output nodes are removed in the sequence representation
        """
        sequences = deepcopy(self.solution)

        for key in self.outputs.keys():
            sequences.pop(key, None)

        return sequences

class Parent(Solution):
    """
    Args:
        gen zero instances
    Returns:
        nothing by default
    Raises:
        no error handling
    """
    # Todo, rows and columns are in params, remove ??
    def __init__(self, params, rows=None, columns=None):
        Solution.__init__(self)
        self.params = params

        self.rows = rows
        self.columns = columns
        self.shape = self.get_shape()

        # minimum is one
        self.levels_back = 1

        # TodO remove ??
        self.statistics = None

        # Todo if rows are 1 then levels back should be same size as columns

        if self.params.user_decisions.levels_back is not None:
            self.levels_back = self.params.user_decisions.levels_back
        elif self.rows == 1:
            self.levels_back = self.columns
        elif self.rows > 1:
            self.levels_back = self.params.randomstate.randint(1, self.columns)

        # ToDo remove?????
        # if self.columns != 1:
        #    self.levels_back = self.params.randomstate.randint(1, self.columns)

        # inputs
        # constants [] + inputs []
        self.constants = self.get_constant_nodes()
        self.inputs = self.get_input_nodes()

        # solution
        # sequence [] + output []
        self.outputs = self.get_output_nodes()
        self.sequences = self.get_sequence_nodes()

        # nodes available
        self.solution = self.get_solution()

        # ToDo: Error in get_active, missing values!!!
        self.active_nodes = self.get_active()

        statistics = NFitness(self.params.user_decisions.fitness_fct,
                              self.params.user_decisions.operators)
        statistics.calc_fitness(self, self.params.problem,
                                self.params.user_decisions.evaluators)

        self.training_fitness = statistics.training_fitness
        self.test_fitness = statistics.test_fitness
        self.complete_solution = self.get_all_nodes()
