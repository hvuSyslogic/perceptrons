"""John Pazarzis, March 25 2017

This code presents a single layer perceptron.
"""
import random
import csv
import math
import functools


def MakePerceptron(filename, learning_rate=0.1,
                   max_iterations=100, verbose=False):
    """Trains a single layer perceptron.

    Arguments:
        filename: The file containing the training data in csv.
        learning_rate: The learning rate.
        max_iterations: The max number of iterations.
        verbose: If true the function will print progress information.

    Returns:
        The perceptron function which receives the input and returns
        the binary classification.
    """
    patterns = _ReadData(filename)
    number_of_patterns = len(patterns)
    if not number_of_patterns:
        raise ImportError
    size_of_pattern = len(patterns[0][0])
    weights = [random.random() for _ in range(size_of_pattern)]
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        global_error = 0.
        for pattern, target in patterns:
            output = _CalculateOutput(weights, pattern)
            local_error = target - output
            for i in range(size_of_pattern):
                weights[i] += learning_rate * local_error * pattern[i]
            global_error += local_error * local_error
        rmse = math.sqrt(global_error / number_of_patterns)
        if verbose:
            print "Iteration {} : RMSE = {}".format(iteration, rmse)
        if global_error == 0:
            break
    if verbose:
        print "Decision boundary: {:+.2f}*x{:+.2f}*y{:+.2f}=0".format(*weights)

    return functools.partial(_CalculateOutput, weights)

def _ReadData(filename, add_bias=True):
    """Reads a csv file containing training data.

    Arguments:
        filename: The file containing the csv data.
        add_bias: If true it will add a input value of 1 for each row.

    Returns:
        A list of lists in the format [pattern, target].
    """
    row = []
    for tokens in csv.reader(open(filename)):
        data = [float(x) for x in tokens]
        pattern = data[0:-1]
        if add_bias:
            pattern.append(1)
        target = data[-1] * 1.
        if target == 0:
            target = -1
        row.append([pattern, target])
    return row

def _ApplyThreshold(value):
    """Threshold function.

    Arguments:
        value: The value to use.

    Returns:
        The corresponding binary value.
    """
    return 1 if value >= 0 else -1

def _CalculateOutput(weights, patern):
    """Calculates the value of a neural.

    Arguments:
        weights: The weights of the synapses.
        patern: The input values.

    Returns:
        The value of the neural.
    """
    assert len(weights) == len(patern)
    return _ApplyThreshold(sum([w * x for w, x in zip(weights, patern)]))
