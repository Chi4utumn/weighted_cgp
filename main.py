# coding=UTF-8
""" Test class for algorithm with cl parameters"""
import argparse
import json
import os

from operations import FunctionGenes
from simplecartesian import SimpleCartesian
from userdecisions import UserDecisions

PARSER = argparse.ArgumentParser()
PARSER.add_argument("-j", "--json", type=str, help="path to json config file")
# ToDo rework verbosity print() methods in code
PARSER.add_argument("-v", "--verbose",
                    action="count", default=0, help="increase output verbosity, ")
PARSER.add_argument("-P", "--problem"
                    , type=str
                    , default='simple_1'
                    , help="available for eval:\
                            simple_1, \
                            gplearn ,\
                            keijzer_n for n in [1-15], \
                            korn_n for n in [1-15], \
                            ")
PARSER.add_argument("-O", "--operators",
                    type=str, default="+, -, *, /,Sin,Cos", help="e.g.: +, -, *, /,Sin,Cos")
PARSER.add_argument("-e", "--evaluator",
                    type=str, default='mse', help="fitness evaluator e.g.: mse ")
PARSER.add_argument("-ee", "--evaluators",
                    type=str, default=None, help="fitness evaluator e.g.: mse ")

#ToDo rename shape c, r, lb
PARSER.add_argument('-c', '--columns',
                    type=int,
                    nargs='+',
                    help="defines possible length of solution candidate")
PARSER.add_argument('-mg', '--max_generations',
                    type=int, default=50, help="stop criterion")
PARSER.add_argument('-ps', '--population_size',
                    type=int, default=200, help="individuals per generation")
PARSER.add_argument('-la', '--es_lambda',
                    type=int, default=199, help="individuals per generation")
PARSER.add_argument('-pm', '--mutation_p',
                    type=float, default=0.15, help="mutation probability")
PARSER.add_argument('-mc', '--mutation_counter',
                    type=int, default=10, help="maximum of mutations for probability")
PARSER.add_argument('-rs', '--random_seed',
                    type=int,
                    default=None,
                    help="if set a defined random seed is used, \
                         if <0 then random 2**32 -1 value is used")
PARSER.add_argument('-C', '--constants',
                    nargs='+',
                    type=int,
                    help="define the number of constant input Nodes with optional range\
                         default # of constants is 0 and range is None\
                         n min max, e.g.: 2 -10 10")
PARSER.add_argument('-log', '--log_params',
                    nargs='+',
                    type=str,
                    help='list of parameters in the order "loglevel",\
                         "storage_option", "logfilepath" ')


ARGS = PARSER.parse_args()

if ARGS.json:
    UD = None
    # ToDo select path of file or relative....
    CWD = os.getcwd()
    PATH = os.path.join(CWD, ARGS.json)
    if os.path.isfile(PATH):
        with open(PATH) as f:
            UD = json.load(f)

        FITNESS_EVAL = UD['fitness_fct']
        EVALUATORS = UD['evaluators']
        ROWS = UD['rows']
        COLUMNS = UD['columns']
        LEVELS_BACK = UD['levels_back']

        OPERATORS = UD['operators']['usedops']
        POPULATION_SIZE = UD['population_size']
        MAX_GENERATIONS = UD['max_generations']
        MUTATION_PROBABILITY = UD['m_probability']
        MUTATION_COUNTER = UD['mutation_counter']
        RANDOM_SEED = UD['random_seed']
        CONSTANT_SIZE = UD['constant_size']
        CONSTANT_RANGE = UD['constant_range']
        ES_LAMBDA = UD['es_lambda']
        VERBOSE = UD['verbose']
        SHOW_PLT = UD['show_plt']
        PROBLEM_NAME = UD['problem']


    else:
        raise FileNotFoundError(ARGS.json)

else:
    FITNESS_EVAL = ARGS.evaluator
    if ARGS.evaluators is not None:
        EVALUATORS = ARGS.evaluators.split(',')
    else:
        EVALUATORS = []
    OPERATORS = ARGS.operators.split(',')
    POPULATION_SIZE = ARGS.population_size
    MAX_GENERATIONS = ARGS.max_generations
    MUTATION_PROBABILITY = ARGS.mutation_p
    MUTATION_COUNTER = ARGS.mutation_counter
    RANDOM_SEED = ARGS.random_seed
    if len(ARGS.constants) == 3:
        CONSTANT_SIZE = ARGS.constants[0]
        CONSTANT_RANGE = ARGS.constants[1:]
    else:
        CONSTANT_SIZE = 0
        CONSTANT_RANGE = None
    ES_LAMBDA = ARGS.es_lambda
    PROBLEM_NAME = ARGS.problem

## LOG Params

if len(ARGS.log_params) == 3:
    LOG_LEVEL = ARGS.log_params[0]
    LOG_STORAGE = ARGS.log_params[1]
    LOG_ALT_PATH = ARGS.log_params[2]

elif len(ARGS.log_params) == 2:
    LOG_LEVEL = ARGS.log_params[0]
    LOG_STORAGE = ARGS.log_params[1]
    LOG_ALT_PATH = None

elif len(ARGS.log_params) == 1:
    LOG_LEVEL = ARGS.log_params[0]
    LOG_STORAGE = None
    LOG_ALT_PATH = None

else:
    LOG_LEVEL = None
    LOG_STORAGE = None
    LOG_ALT_PATH = None


if ARGS.verbose >= 2:
    VERBOSE = True
    SHOW_PLT = True
elif ARGS.verbose == 1:
    VERBOSE = True
    SHOW_PLT = False
else:
    VERBOSE = False
    SHOW_PLT = False


## Shape params
if ARGS.columns is not None:
    if len(ARGS.columns) == 3:
        COLUMNS = ARGS.columns[0]
        ROWS = ARGS.columns[1]
        LEVELS_BACK = ARGS.columns[2]
    elif len(ARGS.columns) == 2:
        COLUMNS = ARGS.columns[0]
        ROWS = ARGS.columns[1]
        LEVELS_BACK = None
    elif len(ARGS.columns) == 1:
        COLUMNS = ARGS.columns[0]
        ROWS = 1
        LEVELS_BACK = None

#Todo rework what is if none is set

## select problem from given params

if VERBOSE:
    # ToDo log in path of execution or in destined folder
    print(os.getcwd())
    print(os.path.dirname(os.path.realpath(__file__)))




# select operators for evolutions
# Todo move to userdecisions
USEDDOPERATORS = FunctionGenes(OPERATORS)


USERPARAMS = UserDecisions(
    population_size=POPULATION_SIZE,
    max_generations=MAX_GENERATIONS,
    operators=USEDDOPERATORS,
    m_probability=MUTATION_PROBABILITY,
    mutation_counter=MUTATION_COUNTER,
    random_seed=RANDOM_SEED,

    logginglevel=LOG_LEVEL,
    output_path=LOG_ALT_PATH,
    data_size=LOG_STORAGE,

    constant_size=CONSTANT_SIZE,
    constant_range=CONSTANT_RANGE,
    rows=ROWS,
    columns=COLUMNS,
    levels_back=LEVELS_BACK,
    es_lambda=ES_LAMBDA,
    problem=PROBLEM_NAME,
    fitness_evaluator=FITNESS_EVAL,
    evaluators=EVALUATORS,
    verbose=VERBOSE,
    show_plt=SHOW_PLT
    )

# SimpleCartesian(Problem=PROBLEM, UserDecisions=USERPARAMS)
SimpleCartesian(UserDecisions=USERPARAMS)
