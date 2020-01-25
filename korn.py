# coding=UTF-8
""" keijzer symbolic regression problems also used in heuristic lab """
import os

import numpy as np

from problem import Problem


class korn_1(Problem):
    '''
    Args:
        name (str): optional default empty string, used for logging
        inputs (int): number of inputs of a problem
        outputs (int): number of outputs of a problem
        data : matrix of problem data

        bibref korns_accuracy_2011
    '''
    def __init__(self,
                 name='korn_1',
                 csv=True,
                 trainsize=10000,
                 min_range=-50,
                 max_range=50):
        """ init korn_1 data instance """
        Problem.__init__(self)
        self.name = name
        self.inputs = 5
        self.outputs = 1


        self.x_n = {3:np.linspace(min_range, max_range, 201)}
        self.y_x = 1.57 + (24.3*self.x_n[3])
        self.title = '1.57 + (24.3*x3)'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_2(Problem):
    """ init korn_2 data instance """
    def __init__(self,
                 csv=True,
                 trainsize=10000,
                 min_range=-50,
                 max_range=50):
        Problem.__init__(self)
        self.name = 'korn_2'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {1:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201),
                    4:np.linspace(min_range, max_range, 201)}
        self.y_x = 0.23 + (14.2*((self.x_n[3]+self.x_n[1])/(3.0*self.x_n[4])))
        self.title = '0.23 + (14.2*((x3+x1)/(3.0*x4)))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_3(Problem):
    """ init korn_3 data instance """
    def __init__(self,
                 csv=True,
                 trainsize=10000,
                 min_range=-50,
                 max_range=50):
        Problem.__init__(self)
        self.name = 'korn_3'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201),
                    1:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201),
                    4:np.linspace(min_range, max_range, 201)}
        self.y_x = -5.41 + (4.9*(((self.x_n[3]-self.x_n[0])+(self.x_n[1]/self.x_n[4]))/(3*self.x_n[4])))
        self.title = '-5.41 + (4.9*(((x3-x0)+(x1/x4))/(3*x4)))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_4(Problem):
    '''
    Args:
        korn_4
        y = -2.3 + (0.13*sin(x2))
    '''
    def __init__(self,
                 csv=True,
                 trainsize=10000,
                 min_range=-50,
                 max_range=50):
        """ init korn_4 data instance """
        Problem.__init__(self)
        self.name = 'korn_4'
        self.inputs = 5
        self.outputs = 1


        self.x_n = {2:np.linspace(min_range, max_range, 201)}
        self.y_x = -2.3 + (0.13*np.sin(self.x_n[2]))
        self.title = 'y = -2.3 + (0.13*sin(x2))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_5(Problem):
    '''
    Args:
        based on f(x4) = 3.0 + (2.13*log(x4))
        korn_5
    '''
    def __init__(self,
                 csv=True,
                 trainsize=10000,
                 min_range=-50,
                 max_range=50):
        """ init korn_5 data instance """
        Problem.__init__(self)
        self.name = 'korn_5'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {4:np.linspace(min_range, max_range, 1000)}
        self.y_x = 3.0 + (2.13*np.log(self.x_n[4]))
        self.title = 'f(x4) = 3.0 + (2.13*log(x4))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_6(Problem):
    '''
    Args:
        based on f(x) = 1.3 + (0.13*sqrt(x0))
        korn_6
    '''
    def __init__(self,
                 csv=True,
                 trainsize=10000,
                 min_range=-50,
                 max_range=50):
        """ init korn_6 data instance """
        Problem.__init__(self)
        self.name = 'korn_6'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 1000)}
        self.y_x = 1.3 + (0.13*np.sqrt(self.x_n[0]))

        self.title = 'f(x) = 1.3 + (0.13*sqrt(x0))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_7(Problem):
    '''
    Args:
        213.80940889 - (213.80940889*exp(-0.54723748542*x0))
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_7 data instance """
        Problem.__init__(self)
        self.name = 'korn_7'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 1000)}
        self.y_x  = 213.80940889 - (213.80940889*np.exp(-0.54723748542*self.x_n[0]))
        self.title = 'y = 213.80940889 - (213.80940889*exp(-0.54723748542*x0))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)

        else:
            raise NotImplementedError

class korn_8(Problem):
    '''
    Args:
        6.87 + (11*sqrt(7.23*x0*x3*x4))
        korn_8
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_8 """
        Problem.__init__(self)
        self.name = 'korn_8'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201),
                    4:np.linspace(min_range, max_range, 201)}
        self.y_x = 6.87 + (11*np.sqrt(7.23*self.x_n[0]*self.x_n[3]*self.x_n[4]))

        self.title = 'f(x) = 6.87 + (11*sqrt(7.23*x0*x3*x4))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_9(Problem):
    '''
    Args:
        korn_9
        f(x0,x1,x2,x3) = ((sqrt(x0)/log(x1))*(exp(x2)/square(x3)))
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_9 """
        Problem.__init__(self)
        self.name = 'korn_9'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201),
                    1:np.linspace(min_range, max_range, 201),
                    2:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201)}
        self.y_x = ((np.sqrt(self.x_n[0])/np.log(self.x_n[1]))*(np.exp(self.x_n[2])/(self.x_n[3]**2)))

        self.title = 'f(x0,x1,x2,x3) = ((sqrt(x0)/log(x1))*(exp(x2)/square(x3)))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_10(Problem):
    '''
    Args:
        korn_10
        0.81 + (24.3*(((2.0*x1)+(3.0*square(x2)))/((4.0*cube(x3))+(5.0*quart(x4)))))
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_10 """
        Problem.__init__(self)
        self.name = 'korn_10'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {1:np.linspace(min_range, max_range, 201),
                    2:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201),
                    4:np.linspace(min_range, max_range, 201)}

        self.y_x = 0.81 + (24.3 *
                           (((2.0 * self.x_n[1]) + (3.0 * (self.x_n[2])**2))
                            /((4.0 * (self.x_n[3]**3)) + (5.0 * (self.x_n[4]**4)))))

        self.title = 'f(x1, x2, x3, x4) = \
                     0.81 + (24.3*(((2.0*x1)+(3.0*square(x2)))/((4.0*cube(x3))+(5.0*quart(x4)))))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_11(Problem):
    '''
    Args:
        korn_11
        6.87 + (11*cos(7.23*x0*x0*x0))
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_11 """
        Problem.__init__(self)
        self.name = 'korn_11'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201)}
        self.y_x = 6.87 + (11*np.cos(7.23*self.x_n[0]*self.x_n[0]*self.x_n[0]))

        self.title = 'f(x0) = 6.87 + (11*cos(7.23*x0*x0*x0))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_12(Problem):
    '''
    Args:
        korn_12
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_12 """
        Problem.__init__(self)
        self.name = 'korn_12'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201),
                    4:np.linspace(min_range, max_range, 201)}
        self.y_x = 2.0 - (2.1*(np.cos(9.8*self.x_n[0])*np.sin(1.3*self.x_n[4])))

        self.title = 'f(x0, x1, x2, x3) = 2.0 - (2.1*(cos(9.8*x0)*sin(1.3*x4)))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_13(Problem):
    '''
    Args:
        korn_13
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_13 """
        Problem.__init__(self)
        self.name = 'korn_13'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201),
                    1:np.linspace(min_range, max_range, 201),
                    2:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201)}
        self.y_x = 32.0 - (3.0 *
                           (np.tan(self.x_n[0])/np.tan(self.x_n[1])) *
                           (np.tan(self.x_n[2])/np.tan(self.x_n[3])))

        self.title = 'f(x0, x1, x2, x3) = 32.0 - (3.0*((tan(x0)/tan(x1))*(tan(x2)/tan(x3))))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_14(Problem):
    '''
    Args:
        korn_14
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_14 """
        Problem.__init__(self)
        self.name = 'korn_14'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201),
                    1:np.linspace(min_range, max_range, 201),
                    2:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201)}
        self.y_x = 22.0 - (4.2 *
                           (np.cos(self.x_n[0]) - np.tan(self.x_n[1]))*
                           (np.tanh(self.x_n[2]) / np.sin(self.x_n[3])))

        self.title = 'f(x0, x1, x2, x3) = 22.0 - (4.2*((cos(x0)-tan(x1))*(tanh(x2)/sin(x3))))'

        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError

class korn_15(Problem):
    '''
    Args:
        korn_15
    '''
    def __init__(self,
                 csv=True,
                 trainsize=100,
                 min_range=-50,
                 max_range=50):
        """ init korn_15 """
        Problem.__init__(self)
        self.name = 'korn_15'
        self.inputs = 5
        self.outputs = 1

        self.x_n = {0:np.linspace(min_range, max_range, 201),
                    1:np.linspace(min_range, max_range, 201),
                    2:np.linspace(min_range, max_range, 201),
                    3:np.linspace(min_range, max_range, 201)}
        self.y_x = 12.0 - (6.0 *
                           (np.tan(self.x_n[0]) / np.exp(self.x_n[1]))*
                           (np.log(self.x_n[2]) - np.tan(self.x_n[3])))

        self.title = 'f(x0, x1, x2, x3) = 12.0 - (6.0*((tan(x0)/exp(x1))*(log(x2)-tan(x3))))'
        if csv:
            name = os.path.join('korn', '{}.csv'.format(self.name))
            self.trainingdata, self.testdata = self.load_csv(name=name,
                                                             trainsize=trainsize)
        else:
            raise NotImplementedError
