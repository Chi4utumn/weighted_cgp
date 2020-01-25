# coding=UTF-8
""" This module has the base operators as well as the defined ones +, -, *, / """

# from abc import ABC, abstractmethod
import numpy as np

class FunctionGenes():
    """ All operations and usage in this class """
    def __init__(self, operators):

        self.arity = {'+':2,
                      '-':2,
                      '*':2,
                      '/':2,
                      'Sin':1,
                      'Cos':1,
                      'Pow2':1,
                      'Sqrt':1,
                      'exp':1,
                      'log':1
                      }

        self.usedops = list(set(self.arity.keys()) & set(operators))
        self.operations = {}

        index = 0
        for key in self.arity:
            if key in self.usedops:

                self.operations[index] = key
                index = index + 1


    def __len__(self):
        return len(self.usedops)


    def use(self, op_n, values, bulk=False):
        """ returns calculated value of node """
        op = self.operations[op_n]
        #Todo change is to ==
        if op is '+':
            if bulk:
                return np.add(values[0], values[1])
            else:
                first = values[0]
                for _e in values[1:]:
                    first = first + _e
                return first

        if op is '-':
            if bulk:
                return np.subtract(values[0], values[1])
            else:
                first = values[0]
                for _e in values[1:]:
                    first = first - _e
                return np.array((first))


        if op is '*':
            if bulk:
                return np.multiply(values[0],values[1])
            else:
                return np.prod(values)

        if op is '/':
            if bulk:
                # denominator is zero
                if any(x == 0 for x in values[1]):
                    nan_arr = np.empty_like(values[0])
                    nan_arr.fill(np.nan)
                    return nan_arr
                else:
                    return np.divide(values[0],values[1])
            else:
                first = values[0]
                for _e in values[1:]:
                    if _e == 0:
                        _e = 0.0000000001
                    first = first / _e
                return first

        if op is 'Sin':
            if bulk:
                return np.sin(values).ravel()
            else:
                first = 0
                for _e in values:
                    first = first + np.sin(_e)
            return first.ravel()

        if op is 'Cos':
            if bulk:
                return np.cos(values).ravel()
            else:
                first = 0
                for _e in values:
                    first = first + np.cos(_e)
                return first

        if op is 'Pow2':
            if bulk:
                result = (values[0].reshape(len(values[0]),len(values)) ** 2)
                return result.ravel()
            else:
                first = 0
                for _e in values:
                    first = first + _e ** 2
                return first

        if op is 'Sqrt':
            if bulk:
                if any([x < 0 for x  in values[0]]):
                    # ToDo doesn't work for none values
                    # negatives = [print(x) for x in values[0] if x < 0]
                    # std_dev = np.std(values[0])
                    # if any([abs(x)>std_dev for x in negatives]):                        
                    #      nan_arr = np.empty_like(values[0].fill(np.nan))
                    #      return nan_arr.reshape(len(values[0]),len(values))
                    # else:
                    #     edited = [0 if x< 0 else x for x in values[0]]
                    #     return np.sqrt(edited.reshape(len(values[0]),len(values)))
                    nan_arr = np.empty_like(values[0])
                    nan_arr.fill(np.nan)
                    return nan_arr # .reshape(len(values[0]),len(values))
                else:
                    result = np.sqrt(values[0].reshape(len(values[0]),len(values)))     
                    return result.ravel()
            else:
                first = 0
                for _e in values:
                    first = first + _e
                return np.sqrt(abs(first))

        if op is 'exp':
            if bulk:
                # np.exp(710) --> inf
                if any(x > 709.78 for x in values[0]):
                    nan_arr = np.empty_like(values[0])
                    nan_arr.fill(np.nan)
                    return nan_arr
                else:
                    result = np.exp(values[0].reshape(len(values[0]),len(values)))
                    return result.ravel()
            else:
                first = 0
                for _e in values:
                    first = first + np.exp(_e)
                return np.sqrt(first)

        if op is 'log':
            if bulk:
                # log 0 = -inf, log < 0 not possible
                if any([x <= 0 for x  in values[0]]):
                    # ToDo doesn't work for none values
                    nan_arr = np.empty_like(values[0])
                    nan_arr.fill(np.nan)
                    return nan_arr
                else:
                    result = np.log(values[0].reshape(len(values[0]),len(values)))
                    return result.ravel()
            first = 0
            for _e in values:
                first = first + np.log(_e)
            return np.sqrt(first)
