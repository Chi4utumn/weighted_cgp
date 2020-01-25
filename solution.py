""" solution class tbd"""

import collections
import os
import platform
from abc import ABC
from itertools import count
from json import JSONEncoder

import numpy as np

from fitness import Fitness

PYDOT = False
try:
    import pydot
    PYDOT = True
except ImportError:
    print("pydot not available on {0}".format(platform.platform()))

#pylint: disable=C0123
class Solution(ABC):
    """
    Args:
        fitness (double): fitness of solution smaller value indicates better fitness
        params (algorithm): makes data from Algorithm available in Solution
                            ( e.g: Userdata, Problemdata,..)
        Operators (operators): allowed operator given by params
        InputNodes
        OutputNodes
        nbr_of_const
        rows (int): rows of the solution matrix
        columns (int): columns of the solution matrix
        levelsback (int): number of nodes the inputs of each node can go
        total_of_nodes (int): nodes available
    Returns:
        nothing by default
    Raises:
        no error handling

    """
    _ids = count(0)

    def __init__(self):
        """ ToDo write info text"""
        self.id = next(self._ids)
        self.params = None

        self.rows = None
        self.columns = None
        self.shape = None

        # minimum is one
        self.levels_back = None

        # inputs
        self.constants = {}
        self.inputs = {}

        # solution
        self.outputs = {}
        self.sequences = {}

        self.solution = None
        self.active_nodes = None
        self.complete_solution = None

        self.training_fitness = None
        self.test_fitness = None

    def get_solution(self):
        """ init a possible solution"""
        # concat of dicts, double keys eliminated
        # ToDo, find out which...
        nodelist = {**self.sequences, **self.outputs}

        return nodelist


    def get_constant_nodes(self):
        """
            returns dict of random constants
            within range length and value as floats
        """
        constants = {}
        if self.params.user_decisions.constant_size > 0:
            c_min = min(self.params.user_decisions.constant_range)
            c_max = max(self.params.user_decisions.constant_range)

            c_size = self.params.user_decisions.constant_size
            _c = ((c_max - c_min) * np.random.random_sample((c_size)) + c_min)
            k = range(_c.size)
            constants = dict(zip(k, _c))

        return constants

    def get_input_nodes(self):
        """ returns dict of constants + # of inputs as dict """
        inputs = self.constants.copy()
        for _i in range(self.params.problem.inputs):
            inputs[_i + len(self.constants)] = None
        return inputs

    def get_output_nodes(self):
        """ returns dict for the number of outputs """
        outputs = {}

        # ToDo where did the -1 go????
        # -1 prevents that the range reaches the output nodes
        available_nodes = len(self.inputs) + self.rows * self.columns

        for _i in range(self.params.problem.outputs):
            outputs[_i + available_nodes] = self.params.randomstate.randint(0, available_nodes - 1)

        return outputs

    def get_sequence_nodes(self):
        """ returns dict of initialized gene sequences """
        sequence = {}
        for _i in range(len(self.inputs), self.rows * self.columns + len(self.inputs)):
            # item = Sequence(nn=_i, params=self)
            sequence[_i] = self.get_value(_i)

        return sequence

    def node_numbers(self):
        """ ref matrix for nodes"""
        s_min = len(self.inputs)
        s_max = s_min + self.rows * self.columns
        nodelist = list(range(s_min, s_max))

        nodes = np.zeros((self.rows, self.columns))

        nodes_trans = [*zip(*nodes)]
        k = 0
        for i, _e in enumerate(nodes_trans):
            for j, _ee in enumerate(_e):
                nodes[j, i] = nodelist[k]
                k = k + 1
        return nodes

    def get_value(self, nn):
        """ method to compute value of sequence  """
        sol = []
        levels = []
        node_number = nn
        function_gene = self.params.randomstate.randint(0,
                                    len(self.params.user_decisions.operators))

        nodes = self.node_numbers()
        sol.append(function_gene)


        nodes_trans = [*zip(*nodes)]
        levels.append(list(range(0, len(self.inputs))))
        for _n in nodes_trans:
            levels.append(_n)

        for i, _e in enumerate(levels):
            for _ee in _e:
                if _ee == node_number:
                    current_level = i
                    break

        if current_level - self.levels_back <= 0:
            low = 0
        else:
            low = current_level - self.levels_back

        high = levels[current_level-1][-1]

        temp1 = self.params.user_decisions.operators
        optype = temp1.operations[function_gene]
        arity = temp1.arity[optype]

        for i in range(arity):

            #random float
            _w = np.random.uniform(0, 1)
            if low == high:
                if low == 0 and high == 0:
                    _c = 0
                else:
                    _c = current_level - self.levels_back
            else:
                _c = np.random.randint(low, high)
            sol.append(_w)
            sol.append(_c)

        # node sequence looks like
        # 0 0.1 1 0.3 4
        return sol

    def get_active(self):
        """ return list of nodes used for solution"""
        # ToDo get active stuck in infinite loop, possible infinite while
        # undone = True
        result = []

        _o = list(self.outputs.values())

        all_used_nodes = {**self.inputs, **self.sequences}

        seek = []
        for _k, _v in self.outputs.items():
            seek.append(_v)
            result.append(_v)

            while seek:
                tmp2 = []
                tmp = all_used_nodes[seek[-1]]
                if seek[-1] >= len(self.inputs):
                    tmp2 = tmp[2::2]
                else:
                    tmp2.append(seek[-1])

                seek.pop()

                for _e in tmp2:
                    if _e not in result:
                        result.append(_e)
                        seek.append(_e)

            # apppend output node index
            result.append(_k)

        result.sort()
        # ToDo markactive nodes
        return result

    def get_shape(self):
        """ returns rows x columns as string """
        if self.rows and self.columns:
            return '{0}x{1}'.format(self.rows, self.columns)
        else:
            return None

    def to_dot_data(self):
        """ create dot data for visualization """
        # pydot needs to be installed and graphviz is needed
        if PYDOT:
            _tr = self.training_fitness[self.params.user_decisions.fitness_fct]
            _te = self.test_fitness[self.params.user_decisions.fitness_fct]
            label = 'fitness train = {0:.3f}, test = {1:.3f} \n  \\l'.format(_tr, _te)

            graph = pydot.Dot(graph_type='digraph', rankdir="LR", label=label, labelloc='t\\l')

            cluster_inputs = pydot.Cluster('inputs', label='inputs')
            cluster_constants = pydot.Cluster('constants', label='constants')
            cluster_gene = pydot.Cluster('functions', label='functions')
            cluster_outputs = pydot.Cluster('outputs', label='outputs')

            needed = self.get_active()

            for _n in needed:
                if _n in self.inputs.keys():
                    if _n in self.constants.keys():
                        cluster_constants.add_node(
                            pydot.Node(str(_n),
                                       label='c{0} : {1:.2f}'.format(_n, self.constants[_n])))
                    else:
                        cluster_inputs.add_node(pydot.Node(str(_n), label='I{0}'.format(_n)))

                if _n in self.sequences.keys():
                    seq = self.sequences[_n][2::2]
                    seq_wgts = self.sequences[_n][1::2]
                    type_seq = self.sequences[_n][0]

                    dbg2 = self.params.user_decisions.operators.operations[type_seq]

                    cluster_gene.add_node(
                        pydot.Node(str(_n),
                                   label='{0} : {1}'.format(_n, dbg2)))

                    for _j, _s in enumerate(seq):
                        a_edge = pydot.Edge(_s, _n, label=str(round(seq_wgts[_j], 2)))
                        # a_edge.label = str(seq[1:][j-1])
                        cluster_gene.add_edge(a_edge)


            for _e in self.outputs.keys():
                cluster_outputs.add_edge(pydot.Edge(self.outputs[_e], str(_e)))
                cluster_outputs.add_node(
                    pydot.Node(str(_e), label='O{0} : {1}'.format(_e, self.outputs[_e])))

            graph.add_subgraph(cluster_constants)
            graph.add_subgraph(cluster_inputs)
            graph.add_subgraph(cluster_gene)
            graph.add_subgraph(cluster_outputs)

            # node contains gene sequence
            # a sequence needs restrains
            # inputs only go backwards outputs only forward
            # levels back restricts backwards
        else:
            graph = None

        # ToDo create full graph
        return graph

    def store_dot_file(self, name=None):
        """stores dot as file"""
        pic = '.png'
        dot = '.dot'

        dot_dir = self.params.user_decisions.log.logfile_path
        pic_dir = self.params.user_decisions.log.logfile_path

        if name is None:
            filename = 'solution_nbr_{0}'.format(str(self.id))
        else:
            filename = '{0}_solution_nbr_{1}'.format(str(name), str(self.id))

        os.path.join(dot_dir, filename + dot)

        if self.params.user_decisions.log.log_data('S'):
            solution_dot = self.to_dot_data()
            solution_dot.write(os.path.join(dot_dir, filename + dot))

        if self.params.user_decisions.log.log_data('M'):
            (graph,) = pydot.graph_from_dot_file(os.path.join(dot_dir, filename + dot))
            graph.write_png(os.path.join(pic_dir, filename + pic))

    def get_all_nodes(self):
        """ concat inputs, sequences and outputs to dict, no duplicates"""
        an0 = {**self.inputs, **self.sequences}
        an1 = {**an0, **self.outputs}
        return an1


    def get_level_allowed(self, item):
        """
        look up method to check which nodes are in the allowed
        range for usage for the next node

        j >= l        n+(j-l)r <= Cij <= n +jr
        j < l         0 <= Cij <= n +jr

        e.g.:
        l0    l1   l2   l3
        i   [[13., 19., 25.],
        p    [14., 20., 26.],
        u    [15., 21., 27.],
        t    [16., 22., 28.],
        s    [17., 23., 29.],
             [18., 24., 30.]]
        """

        nodes = self.node_numbers()
        startnode = 0
        endnode = 0

        if item in self.inputs.keys():
            current_level = 0
            endnode = len(self.inputs)
        else:
            for row in nodes:
                for j, element in enumerate(row):
                    if element == item:
                        current_level = j + 1
                        endnode = nodes[-1][current_level - 1]

        # ToDo levels back is not in effect
        if current_level - self.levels_back <= 0:
            startnode = 0
        else:
            startnode = nodes[0][current_level - self.levels_back-1]

        return startnode, endnode

    def __str__(self):
        """ override self representation: """
        tmp_str = ''
        _od = collections.OrderedDict(sorted(self.complete_solution.items()))

        # ToDo beautify out and inputs
        for _e in _od.values():
            try:
                abcd = [str(x) for x in _e]
                tmp_str = tmp_str + ' '.join(abcd)
            except:
                tmp_str = tmp_str + str(_e)

            tmp_str = tmp_str + ','

        output = str(self.id) + ',' + str(self.training_fitness) + ',' + tmp_str

        return str(self.__class__.__name__) + ',' + output

    def dump(self, name='default'):
        """ stores json of object """
        path = os.path.join(self.params.user_decisions.log.logfile_path,
                            name + '_id{0}.json'.format(self.id))
        json_string = SolutionEncoder().encode(self)
        file = open(path, 'w')
        file.write(json_string)
        file.close()

class SolutionEncoder(JSONEncoder):
    """ parameters which the user may decide to use """
    # https://github.com/PyCQA/pylint/issues/414
    def default(self, object):  # pylint: disable=E0202
        if isinstance(object, (Fitness, Solution)):
            return object.__dict__

        else:
            return None
