import numpy as np

def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def separateByClass(matrix,category):
	separated = {}
	for i in range(len(matrix)):
		vector = matrix[i]
		if (category[i] not in separated):
			separated[category[i]] = []
		separated[category[i]].append(vector)
	return separated

def summarize(matrix):
	state = [(np.mean(attribute), np.std(attribute)) for attribute in \
          zip(*matrix)]
	return state

def nb_train(matrix,category):
	separated = separateByClass(matrix,category)
	state = {}
	for classValue, instances in separated.iteritems():
		state[classValue] = summarize(instances)
	return state

def calculateProbability(x, mu, sd):
	exponent = np.exp(-(x-mu)**2/(2*sd**2))
	return (1 / (np.sqrt(2*np.pi) * sd)) * exponent

def calculateClassProbabilities(state, inputVector):
	probabilities = {}
	for classValue, classstate in state.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classstate)):
			mean, sd = classstate[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, sd)
	return probabilities

def predict(state, inputVector):
	probabilities = calculateClassProbabilities(state, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def nb_test(testSet,state):
	predictions = []
	for i in range(len(testSet)):
		result = predict(state, testSet[i])
		predictions.append(result)
	return predictions

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error

def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
