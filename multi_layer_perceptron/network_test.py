"""Tests for network."""
import unittest
import csv
import network

class Normalizer(object):
  def __init__(self, min_x, max_x):
    self.min_x = min_x
    self.max_x = max_x

  def __call__(self, x):
    return (x - self.min_x) / (self.max_x - self.min_x)


class TestNodes(unittest.TestCase):
  def testInputNodeCreation(self):
    node = network.InputNode()
    self.assertIsNone(node.value)
    self.assertFalse(node.foreward_connections)
    self.assertTrue(isinstance(node, network.Node))
    self.assertTrue(isinstance(node, network.ForwardConnectable))
    self.assertFalse(isinstance(node, network.BackwardConnectable))


  def testHiddenNode(self):
    node = network.HiddenNode()
    self.assertIsNone(node.value)
    self.assertFalse(node.foreward_connections)
    self.assertFalse(node.backward_connections)
    self.assertTrue(isinstance(node, network.Node))
    self.assertTrue(isinstance(node, network.ForwardConnectable))
    self.assertTrue(isinstance(node, network.BackwardConnectable))

  def testOutputNode(self):
    node = network.OutputNode()
    self.assertIsNone(node.value)
    self.assertFalse(node.backward_connections)
    self.assertTrue(isinstance(node, network.Node))
    self.assertFalse(isinstance(node, network.ForwardConnectable))
    self.assertTrue(isinstance(node, network.BackwardConnectable))

  def testNumberOfLayers(self):
    n = network.Network([2, 2, 1])
    self.assertEqual(len(n.layers), 3)

  def testNumberOfNodes(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(number_of_nodes)
    for layer, n in zip(n.layers, number_of_nodes):
      self.assertEqual(len(layer), n)

  def testSetPattern(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(number_of_nodes)
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
    n = network.Network(number_of_nodes)
    pattern = [9.2, 12.12]
    n.SetPattern(pattern)
    n.ClearValues()
    for layer in n.layers:
      hidden_nodes = layer.nodes
      for node in hidden_nodes:
        self.assertIsNone(node.value)

  def testConstuctionOfNodes(self):
    node = network.InputNode()
    self.assertTrue(isinstance(node, network.InputNode))
    node = network.HiddenNode()
    self.assertTrue(isinstance(node, network.HiddenNode))
    node = network.OutputNode()
    self.assertTrue(isinstance(node, network.OutputNode))

    layer = network.InputLayer(2, add_bias=False)
    self.assertEqual(len(layer), 2)
    for node in layer:
      self.assertTrue(isinstance(node, network.InputNode))

    layer = network.HiddenLayer(3, add_bias=False)
    self.assertEqual(len(layer), 3)
    for node in layer:
      self.assertTrue(isinstance(node, network.HiddenNode))

    layer = network.OutputLayer(3)
    self.assertEqual(len(layer), 3)
    for node in layer:
      self.assertTrue(isinstance(node, network.OutputNode))

  def testConnections(self):
    number_of_input_nodes = 2
    number_of_hidden_nodes = 6
    number_of_output_nodes = 12

    number_of_nodes = [
      number_of_input_nodes,
      number_of_hidden_nodes,
      number_of_output_nodes
    ]

    n = network.Network(number_of_nodes)

    input_layer = n.layers[0]
    self.assertEqual(len(input_layer), number_of_input_nodes)
    for node in input_layer.nodes:
      self.assertTrue(isinstance(node, network.InputNode))
      self.assertIsNone(node.value)
      self.assertFalse(hasattr(node, 'backward_connections'))
      self.assertTrue(hasattr(node, 'foreward_connections'))

    for layer in n.layers[1:-1]:
      for hidden_node in layer.nodes:
        self.assertTrue(isinstance(hidden_node, network.HiddenNode))
        self.assertIsNone(hidden_node.value)
        self.assertEqual(
          len(hidden_node.backward_connections),
          number_of_input_nodes
        )
        self.assertEqual(
          len(hidden_node.foreward_connections),
          number_of_output_nodes
        )

        for node in hidden_node.backward_connections.keys():
          self.assertTrue(isinstance(node, network.InputNode))

        for node in hidden_node.foreward_connections.keys():
          self.assertTrue(isinstance(node, network.OutputNode))

    output_layer = n.layers[-1]
    self.assertEqual(len(output_layer), number_of_output_nodes)
    for node in output_layer.nodes:
      self.assertTrue(isinstance(node, network.OutputNode))
      self.assertIsNone(node.value)
      self.assertFalse(hasattr(node, 'foreward_connections'))
      self.assertEqual(len(node.backward_connections), number_of_hidden_nodes)
      for node, connection in node.backward_connections.iteritems():
        self.assertTrue(isinstance(node, network.HiddenNode))
        self.assertTrue(isinstance(connection, network.Connection))
        self.assertTrue(-1 <= connection.weight <= 1)

  def testSetWeight(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(number_of_nodes)
    n.ClearValues()
    hidden_layer = n.layers[1]
    input_layer = n.layers[0]
    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[0], 12.)
    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[1], 8.)
    pattern = [9.2, 12.12]
    n.SetPattern(pattern)
    n.FeedForward(pattern)
    retrieved = hidden_layer.nodes[0].value
    expected = network.Sigmoid(12. * 9.2 + 8. * 12.12)
    self.assertAlmostEqual(retrieved, expected, places=3)

  def testOutputValue(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(number_of_nodes)
    n.ClearValues()
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
    n.FeedForward(pattern)
    retrieved = output_layer.nodes[0].value
    expected = 0.56 * network.Sigmoid(1.31) + 0.71 * network.Sigmoid(2.04)
    expected = network.Sigmoid(expected)

    self.assertAlmostEqual(retrieved, expected, places=3)

  def testSettingTargetValues(self):
    number_of_nodes = [2, 2, 2]
    n = network.Network(number_of_nodes)
    n.ClearValues()

    output_layer = n.layers[2]
    hidden_layer = n.layers[1]
    input_layer = n.layers[0]

    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[0], 0.23)
    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[1], 0.18)
    n.SetWeight(hidden_layer.nodes[1], input_layer.nodes[0], 0.47)
    n.SetWeight(hidden_layer.nodes[1], input_layer.nodes[1], 0.14)

    n.SetWeight(output_layer.nodes[0], hidden_layer.nodes[0], 0.56)
    n.SetWeight(output_layer.nodes[0], hidden_layer.nodes[1], 0.71)

    n.SetWeight(output_layer.nodes[1], hidden_layer.nodes[0], 0.58)
    n.SetWeight(output_layer.nodes[1], hidden_layer.nodes[1], 0.18)

    target_values = [0.12, 9.27]
    n.SetPattern([9.1, 1.2])
    n.SetTargetValues(target_values)
    n.FeedForward([9.1, 1.2])

    output_layer = n.layers[2]

    # Test errors for output layer.
    for output_node, target_value in zip(output_layer.nodes, target_values):
      value = output_node.value

      expected_actual_error = target_value - value
      retrieved_actual_error = output_node.GetActualError()
      self.assertAlmostEqual(retrieved_actual_error,
                             expected_actual_error, places=3)

      expected_error = value * (1 - value) * retrieved_actual_error
      retrieved_error = output_node.GetError()

      self.assertAlmostEqual(retrieved_error,
                             expected_error, places=3)

      # Test errors for hidden layer.
      for hidden_node in n.layers[1].nodes:
        expected_actual_error = 0.
        for ouput_node in output_layer.nodes:
          connection = n.GetConnection(ouput_node, hidden_node)
          expected_actual_error += connection.weight * ouput_node.GetError()

        retrieved_actual_error = hidden_node.GetActualError()
        self.assertAlmostEqual(retrieved_actual_error,
                               expected_actual_error, places=3)
        value = hidden_node.value
        expected_error = value * (1 - value) * retrieved_actual_error
        retrieved_error = hidden_node.GetError()

        self.assertAlmostEqual(retrieved_error,
                               expected_error, places=3)

  def testProcessPattern(self):
    number_of_nodes = [2, 2, 1]
    n = network.Network(number_of_nodes)
    output_layer = n.layers[2]
    hidden_layer = n.layers[1]
    input_layer = n.layers[0]

    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[0], 0.1)
    n.SetWeight(hidden_layer.nodes[0], input_layer.nodes[1], 0.8)
    n.SetWeight(hidden_layer.nodes[1], input_layer.nodes[0], 0.4)
    n.SetWeight(hidden_layer.nodes[1], input_layer.nodes[1], 0.6)

    n.SetWeight(output_layer.nodes[0], hidden_layer.nodes[0], 0.3)
    n.SetWeight(output_layer.nodes[0], hidden_layer.nodes[1], 0.9)

    n.FeedForward([0.35, 0.9])
    n.ProcessPattern([0.35, 0.9], [0.5])
    self.assertAlmostEqual(output_layer.nodes[0].value, 0.69, places=3)

    # Check output error.
    self.assertAlmostEqual(output_layer.nodes[0].GetError(), -0.0406, places=3)

    # Check new weights for output layer.
    connection = n.GetConnection(output_layer.nodes[0], hidden_layer.nodes[0])
    self.assertAlmostEqual(connection.weight, 0.272392, places=3)
    connection = n.GetConnection(output_layer.nodes[0], hidden_layer.nodes[1])
    self.assertAlmostEqual(connection.weight, 0.87305, places=3)

    # Check new hidden layer weights.
    self.assertAlmostEqual(
      n.GetWeight(hidden_layer.nodes[0], input_layer.nodes[0]),
      0.09916,
      places=3
    )

    self.assertAlmostEqual(
      n.GetWeight(hidden_layer.nodes[0], input_layer.nodes[1]),
      0.7978,
      places=3
    )

    self.assertAlmostEqual(
      n.GetWeight(hidden_layer.nodes[1], input_layer.nodes[0]),
      0.3972,
      places=3
    )

    self.assertAlmostEqual(
      n.GetWeight(hidden_layer.nodes[1], input_layer.nodes[1]),
      0.5928,
      places=3
    )

  def Verify(self, n, training_data, limit=0.15):
    for patern, target in training_data:
      predicted = n.Predict(patern)
      for x, y in zip(predicted, target):
        if abs(x - y) > limit:
          return False
    return True

  def RunBinaryClassifier(self, training_data, max_error=0.001):
    success = False
    for i in range(20):
      try:
        n = network.GetTrainedNetwork(
          training_data,
          max_error,
          5000,
          True,
          [2, 3, 2],
          learning_rate=1.
        )
        if self.Verify(n, training_data):
          success = True
          break
      except network.FailedToTrainNetwork:
        pass
    return success

  def testXORWithTwoOutputs(self):
    xor_data = [
      [[1, 1], [0, 1]],
      [[0, 0], [0, 1]],
      [[1, 0], [1, 0]],
      [[0, 1], [1, 0]],
    ]
    self.assertTrue(self.RunBinaryClassifier(xor_data))

    or_data = [
      [[1, 1], [1, 0]],
      [[0, 0], [0, 1]],
      [[1, 0], [1, 0]],
      [[0, 1], [1, 0]],
    ]
    self.assertTrue(self.RunBinaryClassifier(or_data))

    and_data = [
      [[1, 1], [1, 0]],
      [[0, 0], [0, 1]],
      [[1, 0], [0, 1]],
      [[0, 1], [0, 1]],
    ]
    self.assertTrue(self.RunBinaryClassifier(and_data, max_error=0.0002))

  def ReadIrisFile(self, filename, values, normalizer=None):
    patterns = []
    targets = []
    for i, tokens in enumerate(csv.reader(open(filename))):
      if i > 0:
        patterns.append([float(x) for x in tokens[0:-1]])
        targets.append(values[tokens[-1]])
    if not normalizer:
      normalizer = self.MakePatternNormalizer(patterns)
    patterns = [normalizer(p) for p in patterns]
    for pattern in patterns:
      for value in pattern:
        self.assertTrue(0 <= value <=1)

    return [ [p, t] for p, t in zip(patterns, targets)], normalizer

  def MakePatternNormalizer(self, patterns):
    normalizers = []
    for column_index in range(len(patterns[0])):
      x = [row[column_index] for row in patterns]
      normalizers.append(Normalizer(min(x), max(x)))
    return lambda input: [normalizers[i](value) for i, value in enumerate(input)]

  def VerifyIrisClassifier(self):
    values = {
      'versicolor': [1, 0, 0],
      'setosa': [0, 1, 0],
      'virginica': [0, 0, 1],
    }
    _, normalizer = self.ReadIrisFile('iris.csv', values)
    training_data, normalizer = self.ReadIrisFile('training-patterns.csv', values, normalizer=normalizer)
    n = network.GetTrainedNetwork(training_data, 0.0004, 8000, True, [4, 5, 3])
    verifing_data, _ = self.ReadIrisFile('verification-patterns.csv', values, normalizer=normalizer)
    failed, succeeded = 0, 0
    for pattern, target in verifing_data:
      predicted = n.Predict(pattern)
      for x, y in zip(predicted, target):
        if abs(x - y > 0.15):
          failed += 1
        else:
          succeeded += 1
    success_rate = succeeded * 1. / (succeeded + failed)
    return success_rate

  def testIris(self):
    success_rate = 0.
    for _ in range(10):
      success_rate = self.VerifyIrisClassifier()
      if success_rate > 0.9:
        break
    self.assertTrue(success_rate > 0.9)


