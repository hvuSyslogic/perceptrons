"""Tests for network."""
import unittest
import csv
import network


class TestBuildingNetwork(unittest.TestCase):
  def testNumberOfLayers(self):
    n = network.Network(2, 2, 1)
    self.assertEqual(len(n.layers), 3)

  def testNumberOfNodes(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(*number_of_nodes)
    for layer, n in zip(n.layers, number_of_nodes):
      self.assertEqual(len(layer), n)

  def testSetPattern(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(*number_of_nodes)
    pattern = [9.2, 12.12]
    n.SetPattern(pattern)
    input_nodes = n.layers[0].nodes
    for node, value in zip(input_nodes, pattern):
      self.assertEqual(node.value, value)
    for layer in n.layers[1:]:
      hidden_nodes = layer.nodes
      for node in hidden_nodes:
        self.assertIsNone(node.value)

  def testClear(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(*number_of_nodes)
    pattern = [9.2, 12.12]
    n.SetPattern(pattern)
    n.Clear()
    for layer in n.layers:
      hidden_nodes = layer.nodes
      for node in hidden_nodes:
        self.assertIsNone(node.value)

  def testConnections(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(*number_of_nodes)
    input_layer = n.layers[0]
    for node in input_layer.nodes:
      self.assertTrue(isinstance(node, network.InputNode))
      self.assertIsNone(node.value)
      self.assertFalse(hasattr(node, 'connections'))

    for layer in n.layers[1:]:
      for node in layer.nodes:
        self.assertTrue(isinstance(node, network.Neuron))
        self.assertIsNone(node.value)
        self.assertEqual(len(node.connections), 2)
        for connection in node.connections:
          self.assertTrue(-1 <= connection.weight <= 1)

  def testSetWeight(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(*number_of_nodes)
    n.Clear()
    hidden_layer = n.layers[1]
    input_layer = n.layers[0]
    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[0], 12.)
    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[1], 8.)
    pattern = [9.2, 12.12]
    n.SetPattern(pattern)
    retrieved = hidden_layer.nodes[0].GetValue()
    expected = network.Sigmoid(12. * 9.2 + 8. * 12.12)
    self.assertAlmostEqual(retrieved, expected, places=3)

  def testOutputValue(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(*number_of_nodes)
    n.Clear()
    output_layer = n.layers[2]
    hidden_layer = n.layers[1]
    input_layer = n.layers[0]

    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[0], 0.23)
    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[1], 0.18)
    n.SetWeight(hidden_layer.nodes[1], input_layer.nodes[0], 0.47)
    n.SetWeight(hidden_layer.nodes[1], input_layer.nodes[1], 0.14)

    n.SetWeight(output_layer.nodes[0], hidden_layer.nodes[0], 0.56)
    n.SetWeight(output_layer.nodes[0], hidden_layer.nodes[1], 0.71)

    pattern = [3.5, 2.8]
    n.SetPattern(pattern)
    retrieved = output_layer.nodes[0].GetValue()
    expected = 0.56 * network.Sigmoid(1.31) + 0.71 * network.Sigmoid(2.04)
    expected = network.Sigmoid(expected)

    self.assertAlmostEqual(retrieved, expected, places=3)
