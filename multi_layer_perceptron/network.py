import random
import math

def Sigmoid(x):
  return 1 / (1 + math.exp(-x))

class FailedToSetWeight(Exception):
  pass

class InputNode(object):
  def __init__(self):
    self.value = None

  def SetValue(self, value):
    self.value = value

  def GetValue(self):
    return self.value

class Neuron(InputNode):
  def __init__(self):
    super(Neuron, self).__init__()
    self.connections = []

  def Connect(self, other):
    connection = Connection(other)
    self.connections.append(connection)
    return connection

  def GetValue(self):
    if not self.value:
      s = sum([c.node.GetValue() * c.weight for c in self.connections])
      self.value = Sigmoid(s)
    return self.value

class Layer(object):
  node_type = Neuron

  def __init__(self, number_of_nodes):
    self.nodes = [self.node_type() for _ in range(number_of_nodes)]

  def __len__(self):
    return len(self.nodes)

class InputLayer(Layer):
  node_type = InputNode

class Connection(object):
  def __init__(self, node):
    self.node = node
    self.weight = random.uniform(-1., 1.)

class Network(object):
  def __init__(self, *nodes_per_layer):
    self.layers = [InputLayer(nodes_per_layer[0])]
    self.layers.extend([Layer(n) for n in nodes_per_layer[1:]])
    self.connections = {}
    previous_layer = None
    for layer in self.layers:
      if previous_layer:
        for node1 in layer.nodes:
          for node2 in previous_layer.nodes:
            connection = node1.Connect(node2)
            self.connections[node1, node2] = connection
      previous_layer = layer

  def RandomizeWeights(self):
    for layer in self.layers[1:]:
      for node in layer.nodes:
        for connection in node.connections:
          connection.weight = random.uniform(-1., 1.)

  def Clear(self):
    for layer in self.layers:
      for node in layer.nodes:
        node.value = None

  def SetWeight(self, node1, node2, weight):
    connection = self.connections.get((node1, node2))
    if not connection:
      raise FailedToSetWeight
    connection.weight = weight

  def SetPattern(self, pattern):
    input_layer = self.layers[0]
    for node, value in zip(input_layer.nodes, pattern):
      node.SetValue(value)
