import abc
import csv
import random
import math

class PatternReader(object):
  def __init__(self, filename, skip_first_line=False):
    rows = [tokens for tokens in csv.reader(open(filename))]
    if skip_first_line:
      rows = rows[1:]
    targets = {row[-1] for row in rows}
    self.target_map = {}
    self.input_length = len(rows[0]) - 1
    self.target_length = len(targets)
    for i, target_classification in enumerate(targets):
      target = [0] * len(targets)
      target[i] = 1
      self.target_map[target_classification] = target
    self.patterns = [
      ([float(x) for x in row[:-1]], self.target_map[row[-1]])
      for row in rows
    ]

  def GetClassificationFromOutput(self, output):
    if len(output) != self.target_length:
      raise ValueError
    for classification, target in self.target_map.iteritems():
      for s1, s2 in zip(target, output):
        if s1 + s2 >= 1.85:
          return classification
    return 'Not Found'


def Sigmoid(x):
  return 1 / (1 + math.exp(-x))


class FailedToSetWeight(Exception):
  pass


class Node(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(Node, self).__init__()
    self.value = None

class ForwardConnectable(object):
  def __init__(self):
    super(ForwardConnectable, self).__init__()
    self.foreward_connections = {}


class BackwardConnectable(object):
  def __init__(self):
    super(BackwardConnectable, self).__init__()
    self.backward_connections = {}

  def Connect(self, other):
    connection = Connection()
    self.backward_connections[other] = connection
    other.foreward_connections[self] = connection
    return connection

class ErrorGenerator(object):
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(ErrorGenerator, self).__init__()
    self._error = None

  def ClearError(self):
    self._error = None

  @abc.abstractmethod
  def GetActualError(self):
    pass

  def GetError(self):
    if self._error is None:
      value = self.value
      self._error = value * (1. - value) * self.GetActualError()
    return self._error


class InputNode(Node, ForwardConnectable):
  pass


class HiddenNode(ForwardConnectable, BackwardConnectable, ErrorGenerator, Node):
  def GetActualError(self):
    return sum(
      node.GetError() * connection.weight
      for node, connection in self.foreward_connections.iteritems()
    )


class OutputNode(BackwardConnectable, ErrorGenerator, Node):
  def SetTarget(self, target):
    self._target = target

  def GetActualError(self):
    return self._target - self.value


class Layer(object):
  node_type = HiddenNode

  def __init__(self, number_of_nodes, add_bias=True):
    self.nodes = [self.node_type() for _ in range(number_of_nodes)]

  def __len__(self):
    return len(self.nodes)

  def __iter__(self):
    return iter(self.nodes)


class InputLayer(Layer):
  node_type = InputNode


class OutputLayer(Layer):
  node_type = OutputNode

class Connection(object):
  def __init__(self):
    self.weight = random.uniform(-1., 1.)


class Network(object):
  def __init__(self, *nodes_per_layer):
    self.learning_rate = 1.
    self.layers = [InputLayer(nodes_per_layer[0])]
    self.layers.extend([Layer(n) for n in nodes_per_layer[1:-1]])
    self.layers.append(OutputLayer(nodes_per_layer[-1]))
    self.all_backward_connections = {}
    previous_layer = None
    for layer in self.layers:
      if previous_layer:
        for destination_node in layer.nodes:
          for source_node in previous_layer.nodes:
            connection = destination_node.Connect(source_node)
            self.all_backward_connections[destination_node, source_node] = connection
      previous_layer = layer
    self.ClearValues()

  def RandomizeWeights(self):
    for layer in self.layers[1:]:
      for node in layer.nodes:
        for connection in node.backward_connections:
          connection.weight = random.uniform(-1., 1.)

  def ClearValues(self):
    for layer in self.layers:
      for node in layer.nodes:
        node.value = None

  def ClearErrors(self):
    for layer in self.layers[1:]:
      for node in layer.nodes:
        node.ClearError()

  def GetConnection(self, destination_node, source_node):
    return self.all_backward_connections.get((destination_node, source_node))

  def SetWeight(self, destination_node, source_node, weight):
    connection = self.GetConnection(destination_node, source_node)
    if not connection:
      raise FailedToSetWeight
    connection.weight = weight

  def GetWeight(self, destination_node, source_node):
    connection = self.GetConnection(destination_node, source_node)
    if not connection:
      raise FailedToSetWeight
    return connection.weight

  def SetPattern(self, pattern):
    for node, value in zip(self.layers[0].nodes, pattern):
      node.value = value

  def SetTargetValues(self, target):
    for output_node, target_value in zip(self.layers[-1].nodes, target):
      output_node.SetTarget(target_value)

  def FeedForward(self, pattern):
    self.SetPattern(pattern)
    for layer in self.layers[1:]:
      for node in layer.nodes:
        node.value = Sigmoid(sum(n.value * c.weight for n, c in node.backward_connections.iteritems()))

  def Backpropagate(self):
    for layer in reversed(self.layers[1:]):
      for node1 in layer:
        for node2, connection in node1.backward_connections.iteritems():
          connection.weight += self.learning_rate * node1.GetError() * node2.value


  def ProcessPattern(self, pattern, target):
    self.ClearValues()
    self.ClearErrors()
    self.FeedForward(pattern)
    self.SetTargetValues(target)
    self.Backpropagate()

def BuildClassifier(filename, max_epochs=10):
  pattern_reader = PatternReader(filename)
  input_length = pattern_reader.input_length
  target_length = pattern_reader.target_length

  n = Network(input_length, input_length * 2, target_length)
  for _ in range(max_epochs):
    for pattern, target in pattern_reader.patterns:
      n.ProcessPattern(pattern, target)
      # output_layer = n.layers[2]
      # e = 0
      # print len(output_layer)
      # for node in output_layer:
      #   e += node._error* node._error
      #   print node._error,
      # print e
  return n