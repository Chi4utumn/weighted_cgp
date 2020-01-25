# coding=UTF-8
"""  all problems are stored here """
import itertools
import os
import string
from abc import ABC

import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib 
# matplotlib.rc('xtick', labelsize=20) 
# matplotlib.rc('ytick', labelsize=20) 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
# plt.style.use('C:/Users/Chi/.matplotlib/mpl_configdir/stylelib/presentation.mplstyle')
# ToDo: false positives from linter because of self.training and
#       self.testdata initialized as None, but used as numpy array
#       remove comment if pylint can handle this in future
# pylint: disable=E1136

# ToDo inherit numpy randomstate for calculated or splitted data
class Problem(ABC):
    '''
    Args:
        name (str): optional default empty string, used for logging
        inputs (int): number of inputs of a problem
        outputs (int): number of outputs of a problem
        data : matrix of problem data

        if called without args
        inputs  = None
        outputs = None
        data    = None
    Returns:
        nothing by default
    Raises:
        no error handling
    '''
    def __init__(self, randomstate=None):
        """ init default testing Problem or any other data instance """
        self.name = None
        self.title = None # Equation

        self.inputs = None
        self.outputs = None
        self.trainingdata = None
        self.testdata = None
        self.rng = randomstate

        self.x_n = {}
        self.y_x = None

        #Todo uniform problem data
        self.data_from_csv = None
        self.trainsize = None
        self.testsize = None
        self.splitfactor = None

        # Todo use df:
        self.problem_df = None
        self.y_true_test = None
        self.y_true_train = None
        self.path = None

    def get_problem_df(self, training_result, test_result):
        """ merge data to pandas dataframe """

        fields = {'training':[self.trainingdata, training_result],
                  'test':[self.testdata, test_result]}

        df_s = []
        for key, val in fields.items():
            data = val[0]
            name = key
            d_list = []
            d_list_t = []
            d_list.append(list(data[:, -1]))
            d_list_t.append('y_true_' + name)

            for e in range(data.shape[1]):
                if e in self.x_n.keys():
                    d_list.append(list(data[:,e]))
                    d_list_t.append('x_{0}_{1}'.format(e, name))

            if val[1] is not None:
                d_list.append(val[1])
                d_list_t.append('y_pred_{0}'.format(name))


            df_s.append(pd.DataFrame(np.column_stack(d_list), columns=d_list_t))

        use_for_index = df_s[0]

        if df_s[0].shape < df_s[1].shape:
            use_for_index = df_s[1]

        # self.problem_df = pd.concat(df_s, ignore_index=True, axis=use_for_index)
        try:
            self.problem_df = pd.concat(df_s, axis=1).reindex(use_for_index.index)
        except:
            self.problem_df = pd.concat(df_s, ignore_index=True)
        path_csv = os.path.join(self.path, '{0}.csv'.format(self.name))
        self.problem_df.to_csv(path_csv)


        # self.problem_df = pd.concat([train_df, test_df], axis=1).reindex(use_for_index.index)


    def to_csv(self, trainingresult, testresult, path):
        """
        legacy!!!!
            args:
                trainingresult, list of data calculated based on trainingdata
                testresult, list of data calculated based on testdata
                path to store the csv file
        """

        header_train = ['training I{0}'.format(x) for x in range(self.inputs)]
        header_train.extend(['training O{0}'.format(x) for x in range(self.outputs)])
        header_train.append('training result')
        tmp_train = np.column_stack((self.trainingdata, np.array(trainingresult)))
        train_df = pd.DataFrame(tmp_train)
        train_df.columns = header_train

        header_test = ['test I{0}'.format(x) for x in range(self.inputs)]
        header_test.extend(['test O{0}'.format(x) for x in range(self.outputs)])
        header_test.append('test result')
        tmp_test = np.column_stack((self.testdata, np.array(testresult)))
        test_df = pd.DataFrame(tmp_test)
        test_df.columns = header_test

        use_for_index = test_df

        if train_df.shape > test_df.shape:
            use_for_index = train_df


        df_total = pd.concat([train_df, test_df], axis=1).reindex(use_for_index.index)
        path_csv = os.path.join(path, 'old_{0}.csv'.format(self.name))
        df_total.to_csv(path_csv)


    def legacy_scatter(self, trainingresult=None, testresult=None, path=None, show=False):
        """ x,y plot Problem data """
        fig = plt.figure(figsize=(19.2, 10.8))

        # Todo: drop x,y ref in favor of multiplot more fitting
        # only applies if 1 Input --> 1 Output
        if not self.x_n is None and not self.y_x is None:
            plt.plot(self.x_n, self.y_x, linestyle='dashed')

        plt.scatter(self.trainingdata[:, 0],
                    self.trainingdata[:, 1], c='cyan', marker='h', label='training')
        plt.scatter(self.testdata[:, 0], self.testdata[:, 1], c='blue', marker='d', label='test')

        if not testresult is None:
            plt.scatter(self.testdata[:, 0],
                        testresult, c='darkorange', marker='o', label='test_result')
        if not trainingresult is None:
            plt.scatter(self.trainingdata[:, 0],
                        trainingresult, c='red', marker='s', label='training_result')
        plt.legend(loc='upper right')
        plt.xlabel('x values')
        plt.ylabel('y values')
        plt.title(self.name)
        plt.grid()

        if show:
            plt.show()

        if path:
            path = os.path.join(path, '{0}.png'.format(self.name))
            fig.savefig(path)

    def scatter(self):
        """ x,y plot Problem data """
        plt.rcParams.update({'font.size': 22})

        columns_ax = len([x for x in list(self.problem_df.head()) if 'y_' in x])
        rows_ax = len(self.x_n.keys())

        fsize = (19.2 /2  * columns_ax, 10.8 /2 * rows_ax)

        fig, axes = plt.subplots(nrows=rows_ax,
                                 ncols=columns_ax,
                                 sharey=True,
                                 sharex=True,
                                 figsize=fsize)
        fig.suptitle = self.name + ' ' + self.title
        fig.subplots_adjust(top=0.78)

        y_out = [x for x in list(self.problem_df.head()) if 'y_' in x]
        x_in = [x for x in list(self.problem_df.head()) if 'y_' not in x]

        label = ['training', 'training_result', 'test', 'test_result']
        color = ['cyan', 'darkorange', 'blue', 'red']
        marker = ['h', 'd', 'o', 's']

        ## helper for dashed line
        test = [v for k, v in enumerate(list(self.x_n.keys()))]*len(axes)
        for i, row in enumerate(axes):

            if rows_ax <= 1:

                row.set_title(self.name)
                # row = self.problem_df.plot.scatter(x=x_in[i],
                #                             y=y_out[i], legend=True)
                row.plot(self.x_n[test[i]], self.y_x, linestyle='dashed')

                if len(y_out) > 2:
                    x_in = [x_in[0] if 'train' in x else x_in[1] for x in y_out]
                row.scatter(self.problem_df[x_in[i]],
                            self.problem_df[y_out[i]], c=color[i], marker=marker[i], label=label[i])
                row.set_xlabel(x_in[i])
                row.set_ylabel(y_out[i])
                row.grid()
            else:
                sorted(test)
                ## reorder x_in to fit needs
                _m = {k: v for k, v in enumerate(list(self.x_n.keys()))}
                x_in_ord = ['x_{0}_{1}'.format(_m[i], x[7:]) for x in y_out]
                for j, ax in enumerate(row):

                    ax.plot(self.x_n[test[i]], self.y_x, linestyle='dashed')
                    ax.scatter(self.problem_df[x_in_ord[j]],
                               self.problem_df[y_out[j]],
                               c=color[j],
                               marker=marker[j],
                               label=label[j])

                    ax.set_title(label[j])
                    ax.set_xlabel(x_in_ord[j])
                    ax.set_ylabel(y_out[j])
                    ax.grid()


        if self.path is not None:
            path = os.path.join(self.path, 'df_{0}.png'.format(self.name))
            fig.savefig(path)


    def multiscatter(self, trainingresult=None, testresult=None, path=None, show=False):
        """ scatter for n-dimensional data """
        # Todo https://plot.ly/python/line-and-scatter/

        x_letters = string.ascii_lowercase[-3:] + string.ascii_lowercase[:-3]
        y_letters = string.ascii_uppercase[5:] + string.ascii_lowercase[:5]
        x_labels = [x_letters[x] for x in range(self.inputs)]
        y_labels = [y_letters[x] for x in range(self.outputs)]

        _lbls = x_labels.extend(y_labels)

        n = self.inputs + self.outputs
        fontsizes = itertools.cycle([16 + n, 16 + n])
        fig, axes = plt.subplots(n, n, figsize=(19.2 * (n - 1), 10.8 * (n - 1)))# , sharey=True)
        for x, row in enumerate(axes):
            for y, ax in enumerate(row):
                if x == y:
                    x_multi = [self.trainingdata[:, x], self.testdata[:, x]]
                    n_bins = int(np.sqrt(len(self.testdata)) + 0.5)
                    ax.hist(x_multi, n_bins, histtype='bar', label=['train', 'test'])
                    ax.set_xlabel(x_labels[x], fontsize=next(fontsizes))
                    ax.set_ylabel('freq({})'.format(x_labels[x]), fontsize=next(fontsizes))
                else:
                    ax.scatter(self.trainingdata[:, y],
                               self.trainingdata[:, x], c='cyan', marker='h', label='training')
                    ax.scatter(self.testdata[:, y],
                               self.testdata[:, x], c='blue', marker='d', label='test')

                    if not testresult is None:
                        stacked_test_result = np.column_stack((self.testdata[:, :self.inputs],
                                                               testresult))
                        ax.scatter(stacked_test_result[:, y],
                                   stacked_test_result[:, x],
                                   c='darkorange', marker='o', label='test_result')

                    if not trainingresult is None:
                        stacked_training_result = np.column_stack((self.trainingdata[:,
                                                                                     :self.inputs],
                                                                   trainingresult))
                        ax.scatter(stacked_training_result[:, y],
                                   stacked_training_result[:, x],
                                   c='red', marker='s', label='training_result')
                    ax.set_xlabel(x_labels[y], fontsize=next(fontsizes))
                    ax.set_ylabel('{0}({1})'.format(x_labels[x],
                                                    x_labels[y]),
                                  fontsize=next(fontsizes))
                    ax.grid()
                ax.legend()
        if show:
            plt.show()

        if path:
            path = os.path.join(path, 'ms_{0}.png'.format(self.name))
            fig.savefig(path)

       ## 0 012   00 01 02    00 xy xF
       ## 1 012   10 11 12    yx 11 yF
       ## 2 012   20 21 22    Fx Fy 22

    def load_csv(self, name, splitfactor=None, trainsize=None):
        """
        Args
            name,               ...of the dataset
            splitfactor (o),    to split randomly into train and test with given ratio
            trainsize (o),      split data at given point
        """
        ## Path of csv should be relative to current file
        cwd = os.path.dirname(os.path.realpath(__file__))
        # print(os.path.join(cwd, 'data', name))
        if os.path.exists(os.path.join(cwd, 'data', name)):
            data = np.loadtxt(open(os.path.join(cwd, 'data', name), "rb"), delimiter=",", skiprows=1)
            # ToDo variable data split size
            if splitfactor:
                # split_factor = 0.8
                len_test = len(data) - round(len(data) * splitfactor)
                a = self.rng.randint(len(data), size=len_test)

                data_train = np.array([x for i, x in enumerate(data) if i not in a])
                data_test = np.array([x for i, x in enumerate(data) if i in a])
            else:
                data_train = data[:trainsize, :]
                data_test = data[trainsize:, :]
        else:
            raise FileNotFoundError

        trainingdata = np.column_stack((data_train[:, :self.inputs],
                                        data_train[:, self.inputs:]))
        testdata = np.column_stack((data_test[:, :self.inputs],
                                    data_test[:, self.inputs:]))

        return trainingdata, testdata

class GPlearnExample(Problem):
    '''
    Args:
        name (str): optional default empty string, used for logging
        inputs (int): number of inputs of a problem
        outputs (int): number of outputs of a problem
        data : matrix of problem data

        if called without args
        inputs  = 2
        outputs = 1
        data    = based on y = x0² - x1² + x0 -1
                  shape 50x3

        https://gplearn.readthedocs.io/en/stable/examples.html
    Returns:
        nothing by default
    Raises:
        no error handling
    '''
    def __init__(self,
                 csv=False,
                 name='gplearn_example',
                 trainsize=200,
                 testsize=800):
        """ init default testing Problem or any other data instance """
        Problem.__init__(self)
        self.name = name
        self.inputs = 2
        self.outputs = 1

        ## represents problem y = x0**2 - x1**2 + x1 - 1
        if csv:
            raise NotImplementedError
        else:
            size = (trainsize + testsize)
            data = self.rng.uniform(-1, 1, size * 2).reshape(size, 2)
            input_training = data[:trainsize, :]
            input_test = data[trainsize:, :]
            #input_training = self.rng.uniform(-1, 1, 1000).reshape(200, 2)
            output_training = input_training[:, 0]**2 - \
                              input_training[:, 1]**2 + \
                              input_training[:, 1] - 1

            #input_test = self.rng.uniform(-1, 1, 1000).reshape(800, 2)
            output_test = input_test[:, 0]**2 - \
                          input_test[:, 1]**2 + \
                          input_test[:, 1] - 1

        self.trainingdata = np.column_stack((input_training,
                                             output_training))
        self.testdata = np.column_stack((input_test,
                                         output_test))

    def show(self, trainingresult=None, testresult=None, path=None, show=False):
        """
        plot Problem data
        """
        x0 = np.arange(-1, 1, .1)
        x1 = np.arange(-1, 1, .1)
        y_truth = x0**2 - x1**2 + x1 - 1

        fig = plt.figure(figsize=(19.2, 10.8))

        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        ax1.plot(x0, x1, linestyle='dashed')
        ax1.scatter(self.trainingdata[:, 0], self.trainingdata[:, 1], c='red', label='training')
        ax1.scatter(self.testdata[:, 0], self.testdata[:, 1], c='blue', label='test')

        ax2.plot(x0, y_truth, linestyle='dashed')
        ax2.scatter(self.trainingdata[:, 0], self.trainingdata[:, 2], c='red', label='training')
        ax2.scatter(self.testdata[:, 0], self.testdata[:, 2], c='blue', label='test')

        ax3.plot(x1, y_truth, linestyle='dashed')
        ax3.scatter(self.trainingdata[:, 1], self.trainingdata[:, 2], c='red', label='training')
        ax3.scatter(self.testdata[:, 1], self.testdata[:, 2], c='blue', label='test')

        if testresult:
            ax2.scatter(self.testdata[:, 0], testresult, c='darkorange', label='result_test')
            ax3.scatter(self.testdata[:, 1], testresult, c='darkorange', label='result_test')
        if trainingresult:
            ax2.scatter(self.trainingdata[:, 0], trainingresult, c='red', label='result_test')
            ax3.scatter(self.trainingdata[:, 1], trainingresult, c='red', label='result_test')

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        ax3.legend(loc='upper right')

        if show or path:
            plt.show()


    def show3D(self, trainingresult=None, testresult=None, path=None, show=False):
        """
        plot Problem data
        """
        x0 = np.arange(-1, 1, .1)
        x1 = np.arange(-1, 1, .1)
        x0, x1 = np.meshgrid(x0, x1)
        y_truth = x0**2 - x1**2 + x1 - 1

        ax = plt.figure().gca(projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        _surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1, color='green', alpha=0.5)
        _points1 = ax.scatter(self.trainingdata[:, 0],
                              self.trainingdata[:, 1],
                              self.trainingdata[:, 2],
                              c='red', label='training')
        _points2 = ax.scatter(self.testdata[:, 0],
                              self.testdata[:, 1],
                              self.testdata[:, 2],
                              c='blue', label='test')

        if testresult:
            ax.scatter(self.testdata[:, 0],
                       self.testdata[:, 1],
                       testresult, c='darkorange', label='result_test')
        if trainingresult:
            ax.scatter(self.trainingdata[:, 0],
                       self.trainingdata[:, 1],
                       trainingresult, c='red', label='result_training')

        if show or path:
            plt.show()

        # maybe todo: http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
        # if path:
        #    path = os.path.join(path, '{0}.png'.format(self.name))
        #    plt.savefig(path)

# ToDo add class for csv import
class csv_data(Problem):
    """
        import any csv values
    """
    def __init__(self,
                 inputs,
                 outputs,
                 splitfactor=None,
                 name="default_csv",
                 title=None,
                 trainsize=18):
        super().__init__()
        Problem.__init__(self)
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

        # ToDo x,y needed for scatter, --> variable size!!!
        if os.path.exists(os.path.join('data', '{}.csv'.format(self.name))):
            data = np.loadtxt(open(os.path.join('data', '{}.csv'.format(self.name)), "rb"),
                              delimiter=",",
                              skiprows=1)
        self.x = data[0]
        self.y = data[1]

        self.trainingdata, self.testdata = self.load_csv(name='{}.csv'.format(self.name),
                                                         splitfactor=splitfactor,
                                                         trainsize=trainsize)

class simple_1(Problem):
    '''
    Args:
        inputs  = 1 , outputs = 1, based on y = x²
    '''
    def __init__(self, randomstate, name='x_square', trainsize=80, testsize=20):
        """ init default testing Problem or any other data instance """
        Problem.__init__(self, randomstate)
        self.name = name
        self.inputs = 1
        self.outputs = 1

        min_range = -10
        max_range = 10
        self.x = np.linspace(min_range, max_range, 201)
        self.y = self.x ** 2
        #Todo change to expression
        self.title = 'y = x**2' # optional for plot

        input_training = self.rng.uniform(min_range,
                                          max_range,
                                          trainsize * self.inputs).reshape(trainsize, self.inputs)
        output_training = input_training[:, 0] ** 2

        input_test = self.rng.uniform(min_range,
                                      max_range,
                                      testsize * self.inputs).reshape(testsize, self.inputs)
        output_test = input_test[:, 0] ** 2

        self.trainingdata = np.column_stack((input_training,
                                             output_training))
        self.testdata = np.column_stack((input_test,
                                         output_test))
