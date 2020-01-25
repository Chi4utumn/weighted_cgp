# utf-8
""" walk through results and put results to csv"""
import datetime
import json
import os
import shutil
import time
import sys
from functools import reduce
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DataCollector:
    """ search for given folder in path and return valid results in folder structure
        # ./
        #  |-- search directory
        #  |-- data storage
        #       |-- problem_name
        #           |--evaluator
        #               |--foldername
    """
    def __init__(self, sort_path, save_path, result_path):
        self.raw_df = None
        self.search_path = sort_path
        self.result_path = result_path
        self.data_path = save_path
        self.timestamp = datetime.datetime.fromtimestamp(
                time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        if not os.path.exists(os.path.abspath(self.result_path)):
            os.makedirs(os.path.abspath(self.result_path))

        if not os.path.exists(os.path.abspath(self.data_path)):
            os.makedirs(self.data_path)

        # Todo remove
        print("search path:\t{0}".format(self.search_path))
        print("result path:\t{0}".format(self.result_path))
        print("storage path:\t{0}".format(self.data_path))


    def create_csv(self, overwrite_existing=False, name='raw.csv'):
        if overwrite_existing or not os.path.isfile(os.path.join(self.result_path, name)):
            shutil.rmtree(os.path.join(self.result_path, name))

            # if not os.path.isfile(os.path.join(self.result_path, name)):
            self.raw_df = self.get_data_from_files()
            self.raw_df.to_csv(os.path.join(self.result_path, name))
            print('file created:\n{0}'.format(os.path.join(self.result_path, name)))
        else:
            print('file already exist')

    def load_json(self, path_of):
        """ json to data structure """
        with open(path_of) as json_file:
            data = json.load(json_file)
            return data

    def sort_files(self):
        """ moved folders form source to dest """
        if not os.path.exists(self.search_path):
            raise NotADirectoryError
        else:

            root_path = self.search_path
            dest_path = self.data_path
            for root, _dirs, files in os.walk(root_path, topdown=False):

                dirname = os.path.split(root)[-1]
                usr_cfg = 'userconfig.json'
                if usr_cfg in files:

                    _ud = self.load_json(os.path.join(root, usr_cfg))

                    if _ud['random_seed'] is None or 'problem' not in _ud.keys():
                        # move to random dir
                        if not os.path.exists(os.path.join(dest_path, 'random')):
                            os.makedirs(os.path.join(dest_path, 'random'))

                        if not os.path.exists(os.path.join(dest_path, 'random', dirname)):
                            shutil.move(root, os.path.join(dest_path, 'random'))
                        else:
                            shutil.rmtree(root)

                    else:
                        move_path = os.path.join(dest_path, _ud['fitness_fct'], _ud['problem'])
                        if not os.path.exists(move_path):
                            os.makedirs(move_path)

                        if not os.path.exists(os.path.join(move_path, dirname)):
                            shutil.move(root, move_path)
                        else:
                            shutil.rmtree(root)

                else:
                    # discard of folder
                    try:
                        shutil.rmtree(root)
                    except:
                        print('could not remove {0}'.format(root))


    def get_list_of_files(self):
        """ search for two json per folder and return path"""
        file_list = []
        root_path = self.data_path
        for root, _dirs, files in os.walk(root_path, topdown=False):
            if 'random' in root:
                continue
            else:
                logfiles = files

                tmp_files = [os.path.join(root, x) for x in logfiles if x.endswith('json')]

                if len(tmp_files) == 2:
                    file_list.append(tmp_files)

        return file_list


    def get_data_from_files(self):
        data_of_files = []

        todo_files = self.get_list_of_files()
        for _e in todo_files:
            _sol = self.load_json(_e[0])
            _ud = self.load_json(_e[1])

            dict_data = {**_ud, **_sol}

            do_not_log = ['fixed_constant_size', 'limit', 'show_plt',
                          'verbose', 'elitism', 'inputs', 'outputs', 'constants',
                          'sequences', 'solution',
                          'params', 'repro_probability',
                          'x_probability', 'operators', 'evaluators', 'log',
                          'training_fitness', 'test_fitness', 'random_state',
                          'randomstate', 'from_csv', 'trainsize', 'testsize',
                          'splitfactor', 'statistics']

            ### handle operators dict
            ops = {}
            operations = dict_data['operators']['operations']
            arity = {k: v  for k, v in dict_data['operators']['arity'].items() if k in operations.values()}

            ops['operations'] = operations
            ops['arity'] = arity

            dict_data = {**dict_data, **ops}

            ### handle log dict
            log_dict = {'foldername':dict_data['log']['filename'],
                        'timestamp':dict_data['log']['timestamp']
                        }

            dict_data = {**dict_data, **log_dict}

            ### handle fitness
            fitness = {}

            for k, v in dict_data['training_fitness'].items():
                fitness['training_fitness_'+ k] = v

            for k, v in dict_data['test_fitness'].items():
                fitness['test_fitness_'+ k] = v

            ## recalculate uid/checksum of params, save on dict and put result in fitness {}
            if 'uid' not in dict_data.keys():
                fitness['uid'] = self.checksum(dict_data)

            dict_data = {**dict_data, **fitness}

            dict_data = {k:v for k, v in dict_data.items() if k not in do_not_log}

            data_of_files.append(dict_data)

        return pd.DataFrame(data_of_files)


    def checksum(self, param_dict):
        """ create weak uid for given whitelist params """
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
                     'random_seed',
                     'rows']

        hash_dict = {k: v for k, v in param_dict.items() if k in whitelist}

        checksum_key = ''
        for _k, _v in hash_dict.items():
            if _k == 'operators':
                _v = _v['usedops']
            tmp1 = '{0}:{1}'.format(_k, _v)
            if len(checksum_key) > 0:
                checksum_key = '{0},{1}'.format(checksum_key, tmp1)
            else:
                checksum_key = '{0}'.format(tmp1)

        return reduce(lambda x, y: x+y, map(ord, checksum_key))


class DataEvaluator:
    """ search for given folder in path and prepare usable results as csv
    # ./
    #  |-- result directory
    #       |-- csv file of all data
    #       |-- evaluator
                |--...
    """
    def __init__(self, path, name='raw.csv'):

        self.result_path = path
        print("path: {0}".format(self.result_path))

        raw_csv_path = os.path.join(self.result_path, name)
        if not os.path.isfile(os.path.join(self.result_path, name)):
            raise FileNotFoundError
        self.raw_df = pd.read_csv(raw_csv_path)

    def boxplot_by_problem(self, csv_name, store_path):
        test = 'test_fitness_r2_score'
        training = 'training_fitness_r2_score'
        problemname = 'problem'

        self.raw_df = pd.read_csv(os.path.join(store_path, csv_name))
        data_stat = {}

        for _p, p_df in self.raw_df.groupby(problemname):
            name = _p.split('_')
            try:
                _p = '{0}_{1:02d}'.format(name[0], int(name[1]))
            except:
                pass
            fig = plt.figure()
            bp, axes = p_df.boxplot(column=[training, test], by=problemname)
            fig.suptitle(_p)
            plt.savefig(os.path.join(self.result_path, 'py_{0}.png'.format(_p)))
            plt.close(fig)

            data_stat[_p + ' ' + 'training'] = self.get_statistics(p_df[training])
            data_stat[_p + ' ' + 'test'] = self.get_statistics(p_df[test])

        stat_df = pd.DataFrame.from_dict(data_stat)
        stat_df.to_csv(os.path.join(self.result_path, 'box_stat_py.csv'))

    def boxplot_by_param_problem(self, csv_name, store_path):
        test = 'test_fitness_r2_score'
        training = 'training_fitness_r2_score'
        problemname = 'problem'
        uid = 'uid'

        self.raw_df = pd.read_csv(os.path.join(store_path, csv_name))
        data_stat = {}

        for _u, u_df in self.raw_df.groupby(uid):
            data_stat_uid = {}
            for _p, p_df in u_df.groupby(problemname):
                name = _p.split('_')
                try:
                    _p = '{0}_{1:02d}'.format(name[0], int(name[1]))
                except:
                    pass
                fig = plt.figure()
                bp, axes = p_df.boxplot(column=[training, test], by=problemname)
                fig.suptitle(_p)
                plt.savefig(os.path.join(self.result_path, 'py_{0}_{1}.png'.format(_p, _u)))
                plt.close(fig)
                # plt.close(bp)
                bp.clear()
                axes.cla()
                plt.close('all')

                data_stat[_p + ' ' + 'training'] = self.get_statistics(p_df[training])
                data_stat[_p + ' ' + 'test'] = self.get_statistics(p_df[test])
                data_stat_uid[_p + ' ' + 'training'] = self.get_statistics(p_df[training])
                data_stat_uid[_p + ' ' + 'test'] = self.get_statistics(p_df[test])

            stat_df = pd.DataFrame.from_dict(data_stat_uid)
            stat_df.to_csv(os.path.join(self.result_path, 'box_stat_py_{0}_{1}.csv'.format(_p, _u)))

        stat_df = pd.DataFrame.from_dict(data_stat)
        stat_df.to_csv(os.path.join(self.result_path, 'box_stat_py.csv'))

    def get_statistics(self, df):
        data = {}
        count = 'count'
        minimum = 'minimum'
        maximum = 'maximum'
        median = 'median'
        avg = 'average'
        stdev = 'standard deviation'
        var = 'variance'
        q75 = '75th percentile'
        q25 = '25th percentile'
        iqr = 'interquartile range'

        data[count] = len(df)
        data[minimum] = min(df)
        data[maximum] = max(df)
        data[median] = np.median(df)
        data[avg] = np.average(df)
        data[stdev] = np.std(df)
        data[var] = np.var(df)
        nq75, nq25 = np.percentile(df, [75, 25])
        data[q75] = nq75
        data[q25] = nq25
        data[iqr] = nq75 - nq25

        return data

    def prune_r2(self, csv_name, store_path):
        prune_df = pd.read_csv(os.path.join(store_path, csv_name))

        evaluator = 'r2_score'
        sizes = {}
        test = '{0}_fitness_{1}'.format('test', evaluator)
        training = '{0}_fitness_{1}'.format('training', evaluator)

        tmp_df1 = prune_df.loc[(prune_df[test] >= 0) & (prune_df[test] <= 1)]
        tmp_df2 = tmp_df1.loc[(tmp_df1[training] >= 0) & (tmp_df1[training] <= 1)]

        # df = prune_df[(prune_df[[training,test]] >= 0).all(axis=1)]
        # df = prune_df[(prune_df[[training,test]] <= 1).all(axis=1)]
        total = len(prune_df)
        sizes['\# solutions'] = total
        sizes['\# training'] = len(prune_df.loc[(prune_df[test] >= 0) & (prune_df[test] <= 1)])
        sizes['\# test'] = len(prune_df.loc[(prune_df[training] >= 0) & (prune_df[training] <= 1)])

        tmp_df2.to_csv(os.path.join(store_path, '{0}_pruned_x.csv'.format(evaluator)))

        sizes['\# valid'] = len(tmp_df2)

        tmp_df = pd.DataFrame([sizes])
        tmp_df.to_csv(os.path.join(store_path, '{0}_size_zero.csv'.format(evaluator)))

        return tmp_df2

    def get_fitness_for_column(self, param):
        if param in self.raw_df.columns:

            path = os.path.join(self.result_path, 'fitness_vs_{}'.format(param))

            if not os.path.exists(path):
                os.mkdir(path)

            for evaluator, e_df in self.raw_df.groupby('fitness_fct'):
                for column, c_df in e_df.groupby(param):
                    c_df.to_csv(os.path.join(path, '{0}_{1}.csv'.format(evaluator, column)))
        else:
            raise ValueError

    def prune(self, prune_df, evaluator, store_path):

        sizes = {}
        test = '{0}_fitness_{1}'.format('test', evaluator)
        training = '{0}_fitness_{1}'.format('training', evaluator)

        total = len(prune_df)
        sizes['\# solutions'] = total
        sizes['\# training'] = total - prune_df[training].isnull().sum()
        sizes['\# test'] = total - prune_df[test].isnull().sum()

        prune_df[training].replace('', np.nan, inplace=True)
        prune_df[test].replace('', np.nan, inplace=True)

        prune_df.dropna(subset=[training], inplace=True)
        prune_df.dropna(subset=[test], inplace=True)

        prune_df.to_csv(os.path.join(store_path, '{0}_pruned.csv'.format(evaluator)))

        sizes['\# valid'] = len(prune_df)

        tmp_df = pd.DataFrame([sizes])
        tmp_df.to_csv(os.path.join(store_path, '{0}_sizes.csv'.format(evaluator)))

        return prune_df

    def get_fitness(self):
        """ returns fitness sorted by evaluator stores results in result_path """
        path = os.path.join(self.result_path, 'evaluator')

        if not os.path.exists(path):
            os.mkdir(path)

        for evaluator, e_df in self.raw_df.groupby('fitness_fct'):
            e_df.to_csv(os.path.join(path, '{0}.csv'.format(evaluator)))
            pruned = self.prune(e_df, evaluator, path)
            print(evaluator)
            self.store_statistics(pruned, os.path.join(path, '{0}_stat.csv'.format(evaluator)))
            self.store_data(pruned, os.path.join(path, '{0}_data.csv'.format(evaluator)), evaluator)

    def store_statistics(self, dataframe, path):
        """ create df out of df """
        data = {}

        tmp1 = ['training', 'test']
        tmp2 = ['mae', 'mse', 'vaf', 'r2_score']
        data['dataset'] = []
        data['main evaluator'] = []
        data['average'] = []
        data['min'] = []
        data['max'] = []
        data['std'] = []
        data['var'] = []

        for _e in tmp1:
            for f in tmp2:
                data['dataset'].append(_e)
                data['main evaluator'].append(f.replace('_', ' '))

                #+ tmp_list = [float(x.replace(',', '.')) for x in list(dataframe['{0}_fitness_{1}'.format(_e, f)])]
                tmp_list = list(dataframe['{0}_fitness_{1}'.format(_e, f)])

                data['min'].append(np.min(tmp_list))
                data['max'].append(np.max(tmp_list))
                data['average'].append(np.average(tmp_list))
                data['std'].append(np.std(tmp_list))
                data['var'].append(np.var(tmp_list))

        tmp_df = pd.DataFrame(data)
        tmp_df.to_csv(os.path.join(path), index=False)

    def store_data(self, dataframe, path, name):
        """ create df out of df """
        if name in ['mae', 'mse']:
            index = dataframe[['training_fitness_' + name]].idxmin()
        if name in ['r2_score', 'vaf']:
            index = dataframe[['training_fitness_' + name]].idxmax()

        data = {}

        evaluators = ['mae', 'mse', 'vaf', 'r2_score']
        data['evaluator'] = []
        data['training'] = []
        data['test'] = []
        data['remark'] = []

        for evaluator in evaluators:
            data['evaluator'].append(evaluator.replace('_', ' '))
            data['training'].append(float(dataframe['training_fitness_' + evaluator][index]))
            data['test'].append(float(dataframe['test_fitness_' + evaluator][index]))

            if evaluator == name:
                l = len(dataframe['training_fitness_' + evaluator])
                f = dataframe['training_fitness_' + evaluator].isnull().sum()
                t = l - f
                data['remark'].append('${0}/{1}$ solutions found'.format(t, l))
            else:
                data['remark'].append('Secondary evaluator')

        tmp_df = pd.DataFrame(data)
        tmp_df.to_csv(os.path.join(path), index=False)


    def store_full_stat(self):
        """ return matrix of used data """
        data = {}
        total_len = len(self.raw_df)
        data['total'] = [total_len]
        data['problem'] = ['overall']
        evaluators = [e[0] for e in self.raw_df.groupby('fitness_fct')]

        for evaluator in evaluators:
            test = '{0}_fitness_{1}'.format('test', evaluator)
            training = '{0}_fitness_{1}'.format('training', evaluator)

            data[training] = [total_len - self.raw_df[training].isnull().sum()]
            data[test] = [total_len - self.raw_df[test].isnull().sum()]

        for _p, p_df in self.raw_df.groupby('problem'):
            pdf_len = len(p_df)
            data['problem'].append(_p)
            data['total'].append(pdf_len)

            for evaluator in evaluators:
                test = '{0}_fitness_{1}'.format('test', evaluator)
                training = '{0}_fitness_{1}'.format('training', evaluator)

                data[training].append(pdf_len - p_df[training].isnull().sum())
                data[test].append(pdf_len - p_df[test].isnull().sum())


        tmp_df = pd.DataFrame(data)
        tmp_df.to_csv(os.path.join(self.result_path, '{0}_full_stats.csv'.format('pruned')))

    def store_succeeded(self):
        """ return matrix of used data """
        data = {}
        # total_len = len(self.raw_df)
        # data['total'] = [total_len/4]
        # data['problem'] = ['overall']
        data['equation'] = []
        data['total'] = []

        evaluators = [e[0] for e in self.raw_df.groupby('fitness_fct')]

        for evaluator in evaluators:
            test = '{0}_{1}'.format('test', evaluator)
            training = '{0}_{1}'.format('training', evaluator)

            data[training] = []
            data[test] = []

        for _p, p_df in self.raw_df.groupby('problem'):

            pdf_len = len(p_df)
            data['equation'].append("\\ref{{eq:{0}}}".format(_p))
            data['total'].append(pdf_len / 4)

            for _e, e_df in p_df.groupby('fitness_fct'):
                edf_len = len(e_df)
                test = '{0}_fitness_{1}'.format('test', _e)
                training = '{0}_fitness_{1}'.format('training', _e)


                # a = '{:.1%}'.format((edf_len - e_df[training].isnull().sum())/ edf_len)
                data['{0}_{1}'.format('training', _e)].append((edf_len - e_df[training].isnull().sum())/ edf_len)
                data['{0}_{1}'.format('test', _e)].append((edf_len - e_df[test].isnull().sum()) / edf_len)
                # data['total {0}'.format(_e)].append(edf_len)

        data['problem'] = ['{:02d}'.format(int(x[8:-1].split('_')[1]))
                           if x[8:-1].split('_')[0] == 'keijzer'
                           else '{:02d}'.format(int(x[8:-1].split('_')[1]) + 15)
                           for x in data['equation']]

        tmp_df = pd.DataFrame(data)
        tmp_df.columns = [x.replace('_', ' ') for x in list(tmp_df.columns)]
        names = [x.replace('_', ' ') for x in list(tmp_df.columns)]
        new_order = [10,0,1,2,3,4,5,6,7,8,9]
        tmp_df = tmp_df[[names[i] for i in new_order]]
        tmp_df = tmp_df.sort_values('problem')

        tmp_df.to_csv(os.path.join(self.result_path, 'evaluator_succeeded.csv'),
                      encoding='utf-8', index=False)

        data2 = {}
        data2
        data2['min'] = []
        data2['max'] = []
        data2['average'] = []
        for k, v in data.items():
            if 'test' in k or 'training' in k:
                data2['min'].append(min(v))
                data2['average'].append(np.average(v))
                data2['max'].append(max(v))

        tmp_df2 = pd.DataFrame(data2)


        tmp_df2.to_csv(os.path.join(self.result_path, 'min_avg_max_succeeded.csv'),
                       encoding='utf-8', index=False)

    def store_succeeded_eval(self):
        """ return matrix of used data """
        data = {}
        total_len = len(self.raw_df)

        evaluators = [e[0] for e in self.raw_df.groupby('fitness_fct')]

        for evaluator in evaluators:
            test = '{0}_{1}'.format('test', evaluator)
            training = '{0}_{1}'.format('training', evaluator)

            data[training] = []
            data[test] = []

        for _e, e_df in self.raw_df.groupby('fitness_fct'):
            data = {}
            data['problem'] = []
            data['equation'] = []
            data['total'] = []
            data['{0}'.format('training')] = []
            data['{0}'.format('test')] = []

            test = '{0}_fitness_{1}'.format('test', _e)
            training = '{0}_fitness_{1}'.format('training', _e)

            for _p, p_df in e_df.groupby('problem'):

                pdf_len = len(p_df)
                data['equation'].append("\\ref{{eq:{0}}}".format(_p))
                data['total'].append(pdf_len)

                data['{0}'.format('training')].append((pdf_len - p_df[training].isnull().sum())/ pdf_len)
                data['{0}'.format('test')].append((pdf_len - p_df[test].isnull().sum()) / pdf_len)

            data['problem'].extend(['{:02d}'.format(int(x[8:-1].split('_')[1]))
                            if x[8:-1].split('_')[0] == 'keijzer'
                            else '{:02d}'.format(int(x[8:-1].split('_')[1]) + 15)
                            for x in data['equation']])

            tmp_df = pd.DataFrame(data)
            tmp_df = tmp_df.sort_values('problem')
            tmp_df.to_csv(os.path.join(self.result_path, 'p_succeeded_{0}.csv'.format(_e)),
                          encoding='utf-8', index=False)

  

### HeuristicLab eval class
class eval_hl:
    def __init__(self, file_path, file_name):
        self.path = file_path
        self.file_location = os.path.join(self.path, file_name)
        self.data = None
        self.df = None

    def load_txt(self):
        data = []
        with open(self.file_location, 'r') as _f:
            for _l in _f:
                #print(l)
                tmp2 = _l.split('\t')
                tmp = []
                for e in tmp2:
                    if 'Â' in e: # and ('test' in e or 'training' in e):
                        if 'test' in e:
                            e = 'Best R2 test'
                        elif 'training' in e:
                            e = 'Best R2 training'
                        else:
                            e = e.replace('Â', '**')

                    if ('Korn' in e or 'Keijzer' in e) and 'Run' not in e:
                        name = e.split(' ')
                        e = '{0}_{1:02d}'.format(name[0], int(name[1]))

                    try:
                        e = float(e)
                    except:
                        pass
                    tmp.append(e)
                data.append(tmp)
        self.data = data
        self.df = pd.DataFrame(data[1:], columns=data[0])
        self.df.to_csv(os.path.join(self.path, 'hl.csv'), index=False)

    def show_boxplot(self):
        # test = "Best training solution.Pearson's RÂ² (test)"
        # training = "Best training solution.Pearson's RÂ² (training)"
        test = 'Best R2 test'
        training = 'Best R2 training'
        problemname = 'Problem Name'

        boxplot, axes = self.df.boxplot(column=[training, test], by=[problemname], figsize=(19.2, 10.8))
        # plt.setp( axes.xaxis.get_majorticklabels(), rotation=70 )
        # axes.set_title('All problems')
        # axes.set_xlabel('Problem')
        # axes.set_xticks(rotation=90)
        # df = self.df[training]
        # boxplot = df.boxplot(by=problemname)
        # fig = boxplot.get_figure()
        
        # plt.xticks(rotation=90)
        plt.savefig(os.path.join(self.path, 'hl.png'))


    def boxplot_by_problem(self):
        test = 'Best R2 test'
        training = 'Best R2 training'
        problemname = 'Problem Name'

        data_stat = { }

        for _p, p_df in self.df.groupby(problemname):
            name = _p.split(' ')
            try:
                _p = '{0}_{1:02d}'.format(name[0], int(name[1]))
            except:
                pass
            fig = plt.figure()
            bp, axes = p_df.boxplot(column=[training, test], by=problemname)
            fig.suptitle(_p)
            plt.savefig(os.path.join(self.path, 'hl_{0}.png'.format(_p)))
            plt.close(fig)
            bp.clear()
            axes.cla()
            plt.close('all')

            data_stat[_p + ' ' + 'training'] = self.get_statistics(p_df[training])
            data_stat[_p + ' ' + 'test'] = self.get_statistics(p_df[test])

        stat_df = pd.DataFrame.from_dict(data_stat)
        stat_df.to_csv(os.path.join(self.path, 'box_stat_hl.csv'))



    def boxplot_new_problem(self):
        test = 'Best R2 test'
        training = 'Best R2 training'
        problemname = 'Problem Name'
        fig, axes = plt.subplots(nrows=1, ncols=2)
        for _p, p_df in self.df.groupby(problemname):
            name = _p.split(' ')
            try:
                _p = '{0}_{1:02d}'.format(name[0], int(name[1]))
            except:
                pass
            fig = plt.figure()
            p_df.boxplot(column=training, by=problemname, ax=axes[0])
            p_df.boxplot(column=test, by=problemname, ax=axes[1])
            # bp, axes = p_df.boxplot(column=[training, test], by=problemname)
            fig.suptitle(_p)
            plt.savefig(os.path.join(self.path, 'hl_{0}_2.png'.format(_p)))
            plt.close(fig)


    def get_statistics(self, df):
        data = {}
        count = 'count'
        minimum = 'minimum'
        maximum = 'maximum'
        median = 'median'
        avg = 'average'
        stdev = 'standard deviation'
        var = 'variance'
        q75 = '75th percentile'
        q25 = '25th percentile'
        iqr = 'interquartile range'

        data[count] = len(df)
        data[minimum] = min(df)
        data[maximum] = max(df)
        data[median] = np.median(df)
        data[avg] = np.average(df)
        data[stdev] = np.std(df)
        data[var] = np.var(df)
        nq75, nq25 = np.percentile(df, [75 ,25])
        data[q75] = nq75
        data[q25] = nq25
        data[iqr] = nq75 - nq25

        return data


class Result():
    def __init__(self, parameter_list):
        self.raw_data = parameter_list[0]
        self.problem = parameter_list[1]
        self.main_eval = parameter_list[2]
        self.hitquote = self.get_hitquote()
        self.refined_data = self.get_refined()
        pass

    def get_hitquote(self):
        hitquote = 0
        return hitquote


    def get_refined(self):
        refined = None
        return refined


class Results():
    def __init__(self):
        pass
