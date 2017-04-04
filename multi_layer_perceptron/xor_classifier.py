"""implements a XOR classifier
"""
import network

xor_data = [
      [[1, 1], [0]],
      [[0, 0], [0]],
      [[1, 0], [1]],
      [[0, 1], [1]],
    ]


if __name__ == '__main__':
  n = network.GetTrainedNetwork(xor_data, 0.001, 8000, True, [2, 4, 1])

  print (n.Predict([1, 1]))
  print (n.Predict([0, 0]))
  print (n.Predict([1, 0]))
  print (n.Predict([0, 1]))