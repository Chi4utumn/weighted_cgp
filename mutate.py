# coding=UTF-8
"""mutation methods"""
from abc import ABC, abstractmethod
from copy import deepcopy

from candidate import Child


class Mutation(ABC):

    @abstractmethod
    def mutate(self, mutation_candidate):
        """ ToDd explanation """

class PointMutate(Mutation):
    """
        An allele at a randomly chosen gene location is changed to another
        valid random value
            - If a function gene is chosen for mutation, then a valid value is
              the address of any function in the function set
            - If an input gene is chosen for mutation, then a valid value is
              the address of the output of any previous node in the genotype
              or of any program input
            - A valid value for a program output gene is the address of
              the output of any node in the genotype or the address of a program input
            - If a weight is chosen for mutation, a new value is assigned within the
              allowed weight range

        The number of genes that can be mutated in a single application of the
        mutation operator is defined by the user, and is normally a percentage
        of the total number of genes in the genotype.

        The user defined Value is called mutation rate mu_r
        genotype length L_g
        actual number of gene sites that can be mutated mu_g

        mu_g = mu_r * L_g

        miller_cartesian_2011 p.29 ff

        --> good parameter settings for CGP: miller_cartesian_2011 p.31 ff
    """

    def __init__(self, mutation_candidate, RandomState):
        self.user_decisions = mutation_candidate.params.user_decisions
        self.randomstate = RandomState
        self.changed = {}
        self.new_child = Child(mutation_candidate)
        self.tmp_sol = self.mutate(mutation_candidate)

    def mutate(self, mutation_candidate):
        """ apply mutation """

        mutation_p = self.user_decisions.m_probability
        mutation_c = self.user_decisions.mutation_counter

        # succeeded mutation of range of mutation counter
        m_succeeded = 0

        # new_child = CopyCat(self.parent)

        tmp_sol = deepcopy(mutation_candidate.solution)

        start_node = len(self.new_child.inputs)
        lensol = len(self.new_child.solution)
        end_node = len(self.new_child.solution)

        # ToDo create randomstate for mutation
        # ToDo find ou if mutation counter is only +1ned if mutation was successful
        # ToDo what if random selects input or output
        for _i in range(mutation_c):
            prop = self.randomstate.rand()
            if mutation_p >= prop:
                # get random gene to change
                change_idx = self.randomstate.randint(start_node, end_node)
                node = self.new_child.solution[change_idx]

                # if in gene sequence
                if change_idx in self.new_child.sequences.keys():

                    position_for_mutation = self.randomstate.randint(len(node))

                    if position_for_mutation == 0:

                        node_index = 0

                        if self.user_decisions.log.log_level == 'DEBUG':
                            print("from operator use for mutation: " + str(node_index))

                        #change gene to randomly chosen new valid function

                        ops = self.user_decisions.operators.operations
                        ops_len = len(ops)

                        value = self.randomstate.randint(ops_len)
                        arity = self.user_decisions.operators.arity[ops[value]]

                        # arity changes with new operator
                        # function node gets lower arity
                        if(len(node) -1) /2 > arity:
                            node = node[:-2]

                        # function node gets higher arity
                        if(len(node) -1) /2 < arity:
                            # ToDo think of alternative
                            # leads to zero values if minus operator is available
                            node.extend(node[1:])

                        node[node_index] = value

                    #ToDo expand for greater arity
                    if position_for_mutation in [1, 3]:
                        # mutate weight within range
                        #Todo define range

                        new_weight = self.randomstate.rand()
                        node[position_for_mutation] = new_weight

                    #ToDo expand for greater arity
                    if position_for_mutation in [2, 4]:
                        # mutate weight within range
                        #Todo restrict if cartesian is cartesian

                        new_connection = self.randomstate.randint(change_idx)
                        node[position_for_mutation] = new_connection

                else:
                    ##Change output#
                    new_output_connection = self.randomstate.randint(start_node + lensol)
                    node[0] = new_output_connection

                tmp_sol[change_idx] = node
                self.new_child.solution[change_idx] = node
                m_succeeded = m_succeeded + 1

                self.changed[change_idx] = node

        return tmp_sol
