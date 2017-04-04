"""Implements a feed-forward, back propagation perceptron."""

import abc
import random
import math


def Sigmoid(x):
  """Returns the sigmoid value for x.

  Arguments:
    x: The numerical value to calculate its sigmoid.
  """
  return 1 / (1 + math.exp(-x))


class FailedToSetWeight(Exception):
  """Raised when a connection weight cannot be set."""

class FailedToTrainNetwork(Exception):
  """Raised when a the neural network fails to train."""


class Node(object):
  """Generic node serving as the base for the input, hidden and output nodes.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(Node, self).__init__()
    self.value = None

class ForwardConnectable(object):
  """Mixin for nodes that can be connected forward.

  Used for the input and hidden nodes but not for the output which
  cannot be connected forwardly.
  """
  def __init__(self):
    super(ForwardConnectable, self).__init__()
    self.foreward_connections = {}


class BackwardConnectable(object):
  """Mixin for nodes that can be connected backwards.

    Used for the output and hidden nodes but not for the input which
    cannot be connected forwardly.
    """
  def __init__(self):
    super(BackwardConnectable, self).__init__()
    self.backward_connections = {}

  def Connect(self, other):
    """Establishes a connection to another node.

    Note that each node can have backward and forward connections (depending
    on its nature). We keep both to facilitate both the feed forward and the
    back propagation as well.

    Arguments:
       other: The node to connect to.
    """
    connection = Connection()
    self.backward_connections[other] = connection
    other.foreward_connections[self] = connection
    return connection

class ErrorGenerator(object):
  """Mixin used as a base class for nodes that can generate error.

  Only the hidden and output layer can generate errors but not the input.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    super(ErrorGenerator, self).__init__()
    self._error = None

  def ClearError(self):
    """Resets the error."""
    self._error = None

  @abc.abstractmethod
  def GetActualError(self):
    """Must be implemented in derived class.

    Returns:
       The error for the node.
    """

  def GetError(self):
    """Gets the error for the node.

    Calls the GetActualError (which must be implemented in derived levels.

    Returns:
       The error for the node.
    """
    if self._error is None:
      value = self.value
      self._error = value * (1. - value) * self.GetActualError()
    return self._error


class BiasNode(object):
  """Designates a node as bias."""

class InputNode(Node, ForwardConnectable):
  """Used in the input layer and receives the pattern."""

class BiasInputNode(InputNode, BiasNode):
  """Bias node for an input node."""

class HiddenNode(ForwardConnectable, BackwardConnectable, ErrorGenerator, Node):
  """Used in the hidden layer."""

  def GetActualError(self):
    """Implements the ErrorGenerator abstract method.

    Returns:
      The error for the hidden node as it is calculated by the back propagation.
    """
    return sum(
      node.GetError() * connection.weight
      for node, connection in self.foreward_connections.iteritems()
    )

class BiasHiddenNode(HiddenNode, BiasNode):
  """Bias node for a hidden node."""

class OutputNode(BackwardConnectable, ErrorGenerator, Node):
  """Used in the output layer."""

  def SetTarget(self, target):
    """Sets the target value for the node.

    Argument:
      target: (float) the target's value.
    """
    self._target = target

  def GetActualError(self):
    """Implemets the ErrorGenerator abstract method.

    Returns:
      The error for the output node as it is calculated by the back propagation.
    """
    return self._target - self.value

def IsBiasNode(node):
  """Checks wheather a node is bias or not.

  Argument:
    node: The node to check.

  Returns:
     True if the node is bias.
  """
  return isinstance(node, BiasNode)


class Layer(object):
  """The base class for the inpui, hidden and output layers."""

  # The node and biases types to use.
  node_type = None
  bias_node_type = None

  def __init__(self, number_of_nodes, add_bias=True):
    self.nodes = [self.node_type() for _ in range(number_of_nodes)]
    if self.bias_node_type and add_bias:
      self.nodes.append(self.bias_node_type())

  def __len__(self):
    return len(self.nodes)

  def __iter__(self):
    return iter(self.nodes)


class InputLayer(Layer):
  """The Input layer containing input nodes and biases."""
  node_type = InputNode
  bias_node_type = BiasInputNode

class HiddenLayer(Layer):
  """The hidden layer containing hidden nodes and biases."""
  node_type = HiddenNode
  bias_node_type = BiasHiddenNode


class OutputLayer(Layer):
  """The output layer containing hidden nodes and biases."""
  node_type = OutputNode
  bias_node_type = None

class Connection(object):
  """Connects two nodes."""
  def __init__(self):
    self.weight = random.uniform(-1., 1.)


class Network(object):
  def __init__(self, nodes_per_layer, add_bias=False, learning_rate=1.):
    self.learning_rate = learning_rate
    self.layers = [InputLayer(nodes_per_layer[0], add_bias)]
    self.layers.extend([HiddenLayer(n, add_bias) for n in nodes_per_layer[1:-1]])
    self.layers.append(OutputLayer(nodes_per_layer[-1], add_bias=False))
    self.all_backward_connections = {}
    previous_layer = None
    for layer in self.layers:
      if previous_layer:
        for destination_node in layer.nodes:
          if IsBiasNode(destination_node):
            continue
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
    index = 0
    for node in self.layers[0].nodes:
      if IsBiasNode(node):
        node.value = 1
      else:
        node.value = pattern[index]
        index +=1

  def SetTargetValues(self, target):
    for output_node, target_value in zip(self.layers[-1].nodes, target):
      output_node.SetTarget(target_value)

  def FeedForward(self, pattern):
    self.SetPattern(pattern)
    for layer in self.layers[1:]:
      for node in layer.nodes:
        if IsBiasNode(node):
          node.value = 1
        else:
          node.value = Sigmoid(sum(n.value * c.weight for n, c in node.backward_connections.iteritems()))

  def Backpropagate(self):
    for layer in reversed(self.layers[1:]):
      for node1 in layer:
        for node2, connection in node1.backward_connections.iteritems():
          connection.weight += self.learning_rate * node1.GetError() * node2.value

  def GetError(self):
    output_layer = self.layers[-1]
    return sum(abs(node.GetError()) for node in output_layer.nodes) / len(output_layer.nodes)

  def ProcessPattern(self, pattern, target):
    self.ClearValues()
    self.ClearErrors()
    self.FeedForward(pattern)
    self.SetTargetValues(target)
    self.Backpropagate()

  def Predict(self, pattern):
    self.FeedForward(pattern)
    output_layer = self.layers[-1]
    return [node.value for node in output_layer.nodes]

def GetTrainedNetwork(training_data, max_error, max_epochs, add_bias, nodes_per_layer, learning_rate=1.):
  n =Network(nodes_per_layer, add_bias, learning_rate=learning_rate)
  for _ in range(max_epochs):
    for pattern, target in training_data:
      n.ProcessPattern(pattern, target)
      if n.GetError() < max_error:
        return n
  raise FailedToTrainNetwork
