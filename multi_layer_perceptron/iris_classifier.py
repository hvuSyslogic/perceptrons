"""This module can be used to classify iris species.

Instructions to use it from the command line:
  - Start python from the prediction directory.
  - Execute the following statements:

  > from iris_classifier import classifier
  > classifier(4.3, 3, 1.1, 0.1)

  You should see the following output:

  please wait..
  ready to predict..
  'setosa'

  You can use any input from the iris.csv to see how it works.
"""
import csv
import network

values = {
  'versicolor': [1, 0, 0],
  'setosa': [0, 1, 0],
  'virginica': [0, 0, 1],
}

_predictor = None

class Normalizer(object):
  def __init__(self, min_x, max_x):
    self.min_x = min_x
    self.max_x = max_x

  def __call__(self, x):
    return (x - self.min_x) / (self.max_x - self.min_x)


def ReadIrisFile(filename, values, normalizer=None):
  patterns = []
  targets = []
  for i, tokens in enumerate(csv.reader(open(filename))):
    if i > 0:
      patterns.append([float(x) for x in tokens[0:-1]])
      targets.append(values[tokens[-1]])
  if not normalizer:
    normalizer = MakePatternNormalizer(patterns)
  patterns = [normalizer(p) for p in patterns]
  for pattern in patterns:
    for value in pattern:
      assert 0 <= value <= 1
  return [[p, t] for p, t in zip(patterns, targets)], normalizer


def MakePatternNormalizer(patterns):
  normalizers = []
  for column_index in range(len(patterns[0])):
    x = [row[column_index] for row in patterns]
    normalizers.append(Normalizer(min(x), max(x)))
  return lambda input: [normalizers[i](value) for i, value in enumerate(input)]

def MakeIrisPredictor():
  print('please wait..')
  _, normalizer = ReadIrisFile('iris.csv', values)
  training_data, normalizer = ReadIrisFile('iris_training.csv', values, normalizer=normalizer)

  # You can change the layers (meaning the [4, 5, 5, 3] can for example become [4, 5, 3]
  # to see how the net performs when you are running the program. You can try any number
  # of layers / nodes except for the input / output that must always be 4 and 3.
  # Also you can experiment with the minimum error which now is 0.0001.
  n = network.GetTrainedNetwork(training_data, 0.00001, 8000, True, [4, 5, 5, 3])
  print('ready to predict..')
  def Predictor(sepal_length ,sepal_width ,petal_length ,petal_width):
    pattern = [sepal_length ,sepal_width ,petal_length ,petal_width]
    prediction = n.Predict(normalizer(pattern))
    prediction = [0 if x < 0.80 else 1 for x in prediction]
    for key, value in values.iteritems():
      if value == prediction:
        return key
    return 'Not able to predict..'
  return Predictor

def classifier(sepal_length ,sepal_width ,petal_length ,petal_width):
  global _predictor
  if not _predictor:
    _predictor = MakeIrisPredictor()
  return _predictor(sepal_length ,sepal_width ,petal_length ,petal_width)

if __name__ == '__main__':
    predictor = MakeIrisPredictor()

    correct, wrong = 0, 0
    for tokens in csv.reader(open('iris_verifing.csv')):
      try:
        pattern = [float(x) for x in tokens[0:-1]]
        target = tokens[-1]
        if predictor(*pattern) == target:
          correct += 1
        else:
          wrong += 1
        print (predictor(*pattern), target)
      except ValueError:
        pass
    print ('Correct: {} Wrong: {}'.format(correct, wrong))

