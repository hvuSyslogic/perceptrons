"""Tests for perceptron."""
import unittest
import csv
from single_layer import MakePerceptron

class TestSingleLayerPerceptron(unittest.TestCase):
  def testUsingIris(self):
      iris_perceptron = MakePerceptron('iris_training.csv')
      for tokens in csv.reader(open('iris_verify.csv')):
          tokens = [float(x) for x in tokens]
          input = tokens[:-1]
          input.append(1)
          expected = tokens[-1]
          received = iris_perceptron(input)
          self.assertEqual(expected, received)
