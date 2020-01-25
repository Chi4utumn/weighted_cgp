# coding=UTF-8
""" keijzer symbolic regression problems also used in heuristic lab """
import os

import numpy as np

from problem import Problem


class keijzer_1(Problem):
    '''
    Args:
        name (str): optional default empty string, used for logging
        inputs (int): number of inputs of a problem
        outputs (int): number of outputs of a problem
        data : matrix of problem data

        if called without args
        inputs  = 1
        outputs = 1
        data    = based on y = 0.3 * sin(2*Pi *x )
        interval= -1,1
        shape 50x2

        bibref keijzer_genetic_2000
    '''
    def __init__(self,
                 name='keijzer_1',
                 csv=True,
                 trainsize=21,
                 testsize=2022,
                 min_range=-1,
                 max_range=1):
        """ init keijzer_1 data instance """
        Problem.__init__(self)
        self.name = name
        self.inputs = 1
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201)}
        self.y_x = 0.3 * self.x_n[0] * np.sin(2 * np.pi * self.x_n[0])
        self.title = 'y = 0.3 * sin(2*Pi*x)[{0},{1}]'.format(min_range, max_range)

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            input_training = self.rng.uniform(min_range,
                                              max_range,
                                              trainsize * self.inputs).reshape(trainsize,
                                                                               self.inputs)
            output_training = 0.3 * input_training[:, 0] * np.sin(2 * np.pi * input_training[:, 0])

            input_test = self.rng.uniform(min_range,
                                          max_range,
                                          testsize * self.inputs).reshape(testsize, self.inputs)
            output_test = 0.3 * input_test[:, 0] * np.sin(2 * np.pi * input_test[:, 0])

            self.trainingdata = np.column_stack((input_training,
                                                 output_training))
            self.testdata = np.column_stack((input_test,
                                             output_test))

class keijzer_2(keijzer_1):
    """ same keijzer_1 but evaluated in range [-2,2]"""
    def __init__(self, csv=True):
        super().__init__(min_range=-2,
                         max_range=2,
                         trainsize=41,
                         csv=csv,
                         name='keijzer_2')

class keijzer_3(keijzer_1):
    """ same keijzer_1 but evaluated in range [-3,3]"""
    def __init__(self, csv=True):
        super().__init__(min_range=-3,
                         max_range=3,
                         trainsize=61,
                         csv=csv,
                         name='keijzer_3')

class keijzer_4(Problem):
    '''
    Args:
        based on y = x^3 exp(-x) cos(x) sin(x) ( sin^2(x) cos(x) - 1 )
        salustowicz_probabilistic_1997
    '''
    def __init__(self, csv=True, name='', trainsize=201, testsize=401):
        """ init keijzer_4 data instance """
        Problem.__init__(self)
        self.name = 'keijzer_4'
        self.inputs = 1
        self.outputs = 1

        min_range = 0
        max_range = 10.05
        self.x_n = {0:np.linspace(min_range, max_range, 201)}
        self.y_x = self.x_n[0] **3 * \
                 np.exp(-1 * self.x_n[0]) * \
                 np.cos(self.x_n[0]) * \
                 np.sin(self.x_n[0]) * \
                 (np.sin(self.x_n[0])**2 * np.cos(self.x_n[0] ) -1)
        self.title = 'y = x³ exp(-x) cos(x) sin(x) ( sin²(x) cos(x) - 1 )'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            input_training = self.rng.uniform(min_range,
                                              max_range,
                                              trainsize * self.inputs).reshape(trainsize,
                                                                               self.inputs)
            output_training = input_training[:, 0]**3 * \
                              np.exp(-input_training[:, 0]) * \
                              np.cos(input_training[:, 0]) * \
                              np.sin(input_training[:, 0]) * \
                              (np.sin(input_training[:, 0])**2 * \
                               np.cos(input_training[:, 0]) -1)

            input_test = self.rng.uniform(min_range,
                                          max_range,
                                          testsize * self.inputs).reshape(testsize,
                                                                          self.inputs)
            output_test = input_test[:, 0]**3 * \
                          np.exp(-input_test[:, 0]) * \
                          np.cos(input_test[:, 0]) * \
                          np.sin(input_test[:, 0]) * \
                          (np.sin(input_test[:, 0])**2 * \
                           np.cos(input_test[:, 0]) -1)

            self.trainingdata = np.column_stack((input_training,
                                                 output_training))
            self.testdata = np.column_stack((input_test,
                                             output_test))

class keijzer_5(Problem):
    '''
    Args:
        based on f(x,y,z) = (30 *z *x) /  ((x -10)*y²)
        keijzer_5
    '''
    def __init__(self, csv=True, trainsize=1000):
        """ init keijzer_5 data instance """
        Problem.__init__(self)
        self.name = 'keijzer_5'
        self.inputs = 3
        self.outputs = 1

        self.x_n = {0: np.linspace(-1, 1, 1000),
                    1: np.linspace(1, 2, 1000),
                    2: np.linspace(1, 2, 1000)
                   }

        self.y_x = (30 * self.x_n[0] * self.x_n[2]) / ((self.x_n[0] -10) *self.x_n[1] ** 2)
        self.title = 'f(x,y,z) = (30 *z *x) /  ((x -10)*y²)'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_6(Problem):
    '''
    Args:
        based on f(x) = Sum(1/i) | [1,x]
        keijzer_6
    '''
    def __init__(self, csv=True, trainsize=50):
        """ init keijzer_6 data instance """
        Problem.__init__(self)
        self.name = 'keijzer_6'
        self.inputs = 1
        self.outputs = 1

        self.x_n = {0:range(120)}
        self.y_x = []

        for e in self.x_n[0]:
            y = 0
            for i in range(e):
                y = y + 1/(i+1)

            self.y_x.append(y)

        self.title = 'f(x) = Sum(1/i) | [1,x]'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_7(Problem):
    '''
    Args:
        inputs  = 1 , outputs = 1, based on y = log x
        streeter_automated_2003, 5
    '''
    def __init__(self, csv=True, name='', trainsize=100, testsize=20):
        """ init keijzer_7 data instance """
        Problem.__init__(self)
        self.name = 'keijzer_7'
        self.inputs = 1
        self.outputs = 1

        min_range = 1
        max_range = 100
        self.x_n = {0:np.linspace(min_range, max_range, 201)}
        self.y_x = np.log(self.x_n[0])
        self.title = 'y = log(x)'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)

        else:
            input_training = self.rng.uniform(min_range,
                                              max_range,
                                              trainsize * self.inputs).reshape(trainsize,
                                                                               self.inputs)
            output_training = np.log(input_training[:, 0])

            input_test = self.rng.uniform(min_range,
                                          max_range,
                                          testsize * self.inputs).reshape(testsize,
                                                                          self.inputs)
            output_test = np.log(input_test[:, 0])

            self.trainingdata = np.column_stack((input_training,
                                                 output_training))
            self.testdata = np.column_stack((input_test,
                                             output_test))

class keijzer_8(Problem):
    '''
    Args:
        keijzer_8
    '''
    def __init__(self, csv=True, trainsize=101):
        """ init keijzer_8 """
        Problem.__init__(self)
        self.name = 'keijzer_8'
        self.inputs = 1
        self.outputs = 1

        self.x_n = {0:np.linspace(0, 100, 223)}
        self.y_x = np.sqrt(self.x_n[0])

        self.title = 'f(x) = sqrt(x)'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_9(Problem):
    '''
    Args:
        keijzer_9
    '''
    def __init__(self, csv=True, trainsize=101):
        """ init keijzer_9 """
        Problem.__init__(self)
        self.name = 'keijzer_9'
        self.inputs = 1
        self.outputs = 1

        self.x_n = {0:np.linspace(0, 100, 223)}
        self.y_x = np.arcsinh(self.x_n[0])

        self.title = 'f(x) = arcsinh(x)'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_10(Problem):
    '''
    Args:
        keijzer_10
    '''
    def __init__(self, csv=True, trainsize=100):
        """ init keijzer_10 """
        Problem.__init__(self)
        self.name = 'keijzer_10'
        self.inputs = 2
        self.outputs = 1

        self.x_n = {0:np.linspace(0, 1, 223),
                    1:np.linspace(0, 1, 223)}
        self.y_x = self.x_n[0] ** self.x_n[1]

        self.title = 'f(x,y) = x^y'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_11(Problem):
    '''
    Args:
        keijzer_11
    '''
    def __init__(self, csv=True, trainsize=20):
        """ init keijzer_11 """
        Problem.__init__(self)
        self.name = 'keijzer_11'
        self.inputs = 2
        self.outputs = 1

        self.x_n = {0:np.linspace(-3, 3, 223),
                    1:np.linspace(-3, 3, 223)}
        self.y_x = self.x_n[0] ** self.x_n[1] + np.sin((self.x_n[0]-1) * (self.x_n[1]-1))

        self.title = 'f(x,y) = x*y + sin((x-1)(y-1))'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_12(Problem):
    '''
    Args:
        keijzer_12
    '''
    def __init__(self, csv=True, trainsize=20):
        """ init keijzer_12 """
        Problem.__init__(self)
        self.name = 'keijzer_12'
        self.inputs = 2
        self.outputs = 1

        self.x_n = {0:np.linspace(-3, 3, 223),
                    1:np.linspace(-3, 3, 223)}
        self.y_x = self.x_n[0] ** 4 - self.x_n[0]** 3 + self.x_n[1] / (2 - self.x_n[1])

        self.title = 'f(x,y) = x ** 4 - x ** 3 + y / (2 - self.y)'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_13(Problem):
    '''
    Args:
        keijzer_13
    '''
    def __init__(self, csv=True, trainsize=20):
        """ init keijzer_13 """
        Problem.__init__(self)
        self.name = 'keijzer_13'
        self.inputs = 2
        self.outputs = 1

        self.x_n = {0:np.linspace(-3, 3, 223),
                    1:np.linspace(-3, 3, 223)}
        self.y_x = 6 * np.sin(self.x_n[0]) * np.cos(self.x_n[1])

        self.title = 'f(x,y) =  6 * sin(x) * cos(y)'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_14(Problem):
    '''
    Args:
        keijzer_14
    '''
    def __init__(self, csv=True, trainsize=20):
        """ init keijzer_14 """
        Problem.__init__(self)
        self.name = 'keijzer_14'
        self.inputs = 2
        self.outputs = 1

        self.x_n = {0:np.linspace(-3, 3, 223),
                    1:np.linspace(-3, 3, 223)}
        self.y_x = 8 / (2 + self.x_n[0]**2 + self.x_n[1]**2)

        self.title = 'f(x,y) =  8 / (2 + x**2 + y**2)'

        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class keijzer_15(Problem):
    '''
    Args:
        keijzer_15
    '''
    def __init__(self, csv=True, trainsize=20):
        """ init keijzer_15 """
        Problem.__init__(self)
        self.name = 'keijzer_15'
        self.inputs = 2
        self.outputs = 1

        self.x_n = {0:np.linspace(-3, 3, 223),
                    1:np.linspace(-3, 3, 223)}
        self.y_x = (self.x_n[0] ** 3 / 5) + (self.x_n[1]**3 / 2) - self.x_n[1] - self.x_n[0]

        # ToDo: Zero Division in target function
        self.title = 'f(x,y) = self.x ** 3 / 5 + self.y**3 / 2 - self.y -self.x'
        if csv:
            name = os.path.join('keijzer', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError
