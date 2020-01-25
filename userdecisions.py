# coding=UTF-8
# introduction_to_gp.pdf
""" parameters which the user may decide to use """
import datetime
import logging
import os
import time
from functools import reduce
from json import JSONEncoder
from zipfile import ZipFile

import numpy as np

from keijzer import (keijzer_1, keijzer_2, keijzer_3, keijzer_4, keijzer_5,
                     keijzer_6, keijzer_7, keijzer_8, keijzer_9, keijzer_10,
                     keijzer_11, keijzer_12, keijzer_13, keijzer_14,
                     keijzer_15)
from korn import (korn_1, korn_2, korn_3, korn_4, korn_5, korn_6, korn_7,
                  korn_8, korn_9, korn_10, korn_11, korn_12, korn_13, korn_14,
                  korn_15)
from operations import FunctionGenes
from problem import GPlearnExample, csv_data, simple_1


class UserDecisions():
    """ parameters which the user may decide to use """
    def __init__(self,
                 population_size=None,
                 max_generations=42,
                 es_lambda=4,
                 notation='plus',
                 x_probability=None,
                 m_probability=None,
                 mutation_counter=None,
                 reproduction_p=None,
                 operators=None,
                 random_seed=None,

                 constant_size=0,
                 constant_range=[-1, 1],
                 rows=1,
                 columns=23,
                 levels_back=17,

                 problem="simple_1",
                 problem_inputs=1,
                 problem_outputs=1,
                 from_csv=True,
                 trainsize=None,
                 testsize=None,
                 splitfactor=None,

                 fitness_evaluator='mae',
                 evaluators=[],

                 logginglevel=None,
                 output_path=None,
                 data_size=None,

                 show_plt=False,
                 verbose=None
                 ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.m_probability = m_probability
        self.mutation_counter = mutation_counter

        self.operators = operators
        self.constant_range = constant_range
        self.randomstate = np.random
        if random_seed is None:
            self.random_seed = random_seed
        elif random_seed < 0:
            # randint takes signed int value and
            # as error on 32bit python if greater than 31
            self.random_seed = np.random.randint(2**31 - 1)

        else:
            self.random_seed = random_seed

        self.randomstate.seed(self.random_seed)
        self.random_state = self.randomstate.RandomState().get_state()
        self.constant_size = constant_size
        self.rows = rows
        self.columns = columns
        self.levels_back = levels_back

        # The (1+4) evolutionary strategy
        # 1 + λ evolutionary algorithm
        self.es_mu = 1
        self.es_lambda = es_lambda

        # evaluator for algorithm 95p, mean, stddev, variance, pearson, chisquare, mae, mse
        self.fitness_fct = fitness_evaluator
        # list of evaluators to additionally calc but not used for eval
        self.evaluators = evaluators
        if self.fitness_fct not in self.evaluators:
            self.evaluators.append(self.fitness_fct)

        ## SR Problem params
        self.problem = problem
        self.problem_inputs = problem_inputs
        self.problem_outputs = problem_outputs
        self.from_csv = from_csv
        self.trainsize=trainsize
        self.testsize = testsize
        self.splitfactor=splitfactor

        ## verbosity and debug params
        self.show_plt = show_plt
        self.verbose = verbose


        ## depraced
        # self.fixed_constant_size = fixed_constant_size

        ## not implemented
        self.notation = notation
        self.elitism = False
        self.repro_probability = reproduction_p
        self.x_probability = x_probability
        self.limit = 100 #  limit===Ln is the user-determined upper bound of the number of nodes
        self.algorithm = "Weighted CGP"
        self.fitness_strategy = "best"

        self.uid = self.checksum()

        self.log = Log(log_level=logginglevel,
                       output_path=output_path,
                       data_size=data_size,
                       fitness_fct=self.fitness_fct,
                       uid=self.uid,
                       problem=problem)


    def reset_seed(self):
        self.randomstate.seed(self.random_seed)

    def set_random_state(self, rd_state):
        self.random_state = rd_state

    def checksum(self):
        to_hash = dict(iter(self))

        #blacklist = ['randomstate', 'verbose', 'show_plt', 'uid', 'randomstate']
        ## composed of paramters given be the user
        # e.g.: levels back is calculated at runtime and therefore not adjustable
        whitelist = ['columns',
                     'constant_range',
                     'constant_size',
                     'es_lambda',
                     'es_mu',
                     'fitness_fct',
                     'm_probability',
                     'max_generations',
                     'mutation_counter',
                     'operators',
                     'population_size',
                     'problem',
                     'rows']

        hash_dict = { k: v for k, v in to_hash.items() if k in whitelist }

        a = ''
        for k, v in hash_dict.items():
            if k == 'operators':
                v = v.usedops
            b = '{0}:{1}'.format(k,v)
            if len(a) > 0:
                a = '{0},{1}'.format(a,b)
            else:
                a = '{0}'.format(b)

        return reduce(lambda x,y:x+y, map(ord, a))

    def reset_problem(self, problem):
        self.problem = problem.name
        self.problem_inputs = problem.inputs
        self.problem_outputs = problem.outputs

        #Todo uniform problem data
        self.from_csv = problem.data_from_csv
        self.trainsize = problem.trainsize
        self.testsize = problem.testsize
        self.splitfactor = problem.splitfactor
        
    def get_problem(self, name=None):
        """ select problem from given string """
        # Todo pass random to problem init
        if name is None:
            name = self.problem

        # Todo pass parameter
        if name == 'keijzer_1':
            problem = keijzer_1()
        elif name == 'keijzer_2':
            problem = keijzer_2()
        elif name == 'keijzer_3':
            problem = keijzer_3()
        elif name == 'keijzer_4':
            problem = keijzer_4()
        elif name == 'keijzer_5':
            problem = keijzer_5()
        elif name == 'keijzer_6':
            problem = keijzer_6()
        elif name == 'keijzer_7':
            problem = keijzer_7()
        elif name == 'keijzer_8':
            problem = keijzer_8()
        elif name == 'keijzer_9':
            problem = keijzer_9()
        elif name == 'keijzer_10':
            problem = keijzer_10()
        elif name == 'keijzer_11':
            problem = keijzer_11()
        elif name == 'keijzer_12':
            problem = keijzer_12()
        elif name == 'keijzer_13':
            problem = keijzer_13()
        elif name == 'keijzer_14':
            problem = keijzer_14()
        elif name == 'keijzer_15':
            problem = keijzer_15()
        elif name == 'korn_1':
            problem = korn_1()
        elif name == 'korn_2':
            problem = korn_2()
        elif name == 'korn_3':
            problem = korn_3()
        elif name == 'korn_4':
            problem = korn_4()
        elif name == 'korn_5':
            problem = korn_5()
        elif name == 'korn_6':
            problem = korn_6()
        elif name == 'korn_7':
            problem = korn_7()
        elif name == 'korn_8':
            problem = korn_8()
        elif name == 'korn_9':
            problem = korn_9()
        elif name == 'korn_10':
            problem = korn_10()
        elif name == 'korn_11':
            problem = korn_11()
        elif name == 'korn_12':
            problem = korn_12()
        elif name == 'korn_13':
            problem = korn_13()
        elif name == 'korn_14':
            problem = korn_14()
        elif name == 'korn_15':
            problem = korn_15()
        elif name == 'gplearn':
            problem = GPlearnExample()
        elif name == 'simple_1':
            problem = simple_1(self.randomstate)
        # Todo add path, inputs and outputs
        elif name == 'csv_data':
            problem = csv_data(inputs=1, outputs=1)
        else:
            raise Exception
        return problem


    def export_json(self, name=None):
        """ stores json of object """
        if name is None:
            name = 'userconfig.json'
        path = os.path.join(self.log.logfile_path, name)
        json_string = UserDecisionsEncoder().encode(self)
        file = open(path, 'w')
        file.write(json_string)
        file.close()

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __str__(self):
        """ override self representation: """
        ops = ''
        ops = ops + ' - ' + str(self.operators.operations) + '\n'

        return str(self.__class__.__name__) + ':\n' +       \
                'Random Seed\t\t' + str(self.random_seed) + '\n' + \
                'Population Size:\t' + str(self.population_size) + '\n' + \
                'Max Generations:\t' + str(self.max_generations) + '\n' + \
                'crossover propability:\t' + str(self.x_probability) + '\n' + \
                'mutation propability:\t' + str(self.m_probability) + '\n' + \
                'mutation counter:\t' + str(self.mutation_counter) + '\n' + \
                'operators:\n' + ops

class Log():
    """
     Args:
        log_level (str): currently Info and Debug in use, default INFO
        logfile_name (str): name may be set by user, default None

        wraps log file with timestamp + filename in folder
        timestamp: '%Y-%m-%d_%H-%M-%S'
        ./log
           |-- timestamp + filename
                | datetime.log
                | fittestplot.png

        Log_Level 	Numeric value
        CRITICAL 	50
        ERROR 	    40
        WARNING 	30
        INFO 	    20
        DEBUG 	    10
        NOTSET 	    0

    Returns:
        nothing by default
    Raises:
        no error handling
    """
    def __init__(self, problem, fitness_fct, uid, log_level='INFO' ,data_size='XXL', output_path=None):


        self.log_level = log_level
        self.timestamp = datetime.datetime.fromtimestamp(
            time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        self.data_size = data_size
        self.filename = '{0}_{1}_{2}'.format(fitness_fct, uid, self.timestamp)

        if log_level is None:
            self.log_level = 'INFO'

        if data_size is None:
            self.data_size = 'M'

        ## optional root dir as log path
        if output_path is None:
            rel_start = os.path.dirname(os.path.realpath(__file__))
        else:
            rel_start = '.'
        parent_folder = 'log'
        sub_folder = problem

        # Check if root dir is available and problem dir is available
        if not os.path.exists(os.path.join(rel_start, parent_folder)):
            os.makedirs(os.path.join(rel_start, parent_folder))
        self.log_dir = os.path.join(rel_start, parent_folder, sub_folder)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create other dirs:
        self.logfile_path = os.path.join(self.log_dir, self.filename)
        if not os.path.exists(self.logfile_path):
            os.makedirs(self.logfile_path)

        logfile_name = os.path.join(self.logfile_path, 'gcp.log')


        logging.basicConfig(filename=logfile_name,
                            level=self.log_level, format='%(asctime)s %(levelname)s:%(message)s')

        logging.info(logfile_name)

    def log_data(self, log_size):
        sizedict = {'S':10,
                    'M':20,
                    'L':30,
                    'XL':40,
                    'XXL':50,
                    'XXXL':60
                    }
        return sizedict[self.data_size] >= sizedict[log_size]



    def log_info(self, text):
        """ make info log available """
        logging.info(text)


    def log_debug(self, text):
        """ make debug log available """
        logging.debug(text)


    def zip_log(self):
        """ Source: https://www.geeksforgeeks.org/working-zip-files-python/ """
        # path to folder which needs to be zipped
        directory = self.logfile_path

        # calling function to get all file paths in the directory
        file_paths = self.get_all_file_paths(directory)
        zip_path = os.path.join(self.log_dir, self.filename + '.zip')

        # writing files to a zipfile
        with ZipFile(zip_path, 'w') as zip:
            # writing each file one by one
            for file in file_paths:
                zip.write(file)


    def get_all_file_paths(self, directory):
        """ check directories for missing log folders"""
        # initializing empty file paths list
        file_paths = []

        # crawling through directory and subdirectories
        for root, directories, files in os.walk(directory): # pylint: disable=W0612
            for filename in files:
                # join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)

        # returning all file paths
        return file_paths

class UserDecisionsEncoder(JSONEncoder):
    """ parameters which the user may decide to use """
    # https://github.com/PyCQA/pylint/issues/414
    def default(self, object):  # pylint: disable=E0202
        if isinstance(object, (UserDecisions, FunctionGenes, Log)):
            return object.__dict__

        else:
            return None
