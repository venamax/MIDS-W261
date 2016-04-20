from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.linalg import SparseVector
from collections import defaultdict
import hashlib
import numpy as np
from math import exp #  exp(-t) = e^-t
from math import log
import os

def hashFunction(numBuckets, rawFeats, printMapping=False):
    """Calculate a feature dictionary for an observation's features based on hashing.

    Note:
        Use printMapping=True for debug purposes and to better understand how the hashing works.

    Args:
        numBuckets (int): Number of buckets to use as features.
        rawFeats (list of (int, str)): A list of features for an observation.  Represented as
            (featureID, value) tuples.
        printMapping (bool, optional): If true, the mappings of featureString to index will be
            printed.

    Returns:
        dict of int to float:  The keys will be integers which represent the buckets that the
            features have been hashed to.  The value for a given key will contain the count of the
            (featureID, value) tuples that have hashed to that key.
    """
    mapping = {}
    for ind, category in rawFeats:
        featureString = category + str(ind)
        mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(), 16) % numBuckets)
    if(printMapping): print mapping
    sparseFeatures = defaultdict(float)
    for bucket in mapping.values():
        sparseFeatures[bucket] += 1.0
    return dict(sparseFeatures)

def parseHashPoint(point, numBuckets):
    """Create a LabeledPoint for this observation using hashing.

    Args:
        point (str): A comma separated string where the first value is the label and the rest are
            features.
        numBuckets: The number of buckets to hash to.

    Returns:
        LabeledPoint: A LabeledPoint with a label (0.0 or 1.0) and a SparseVector of hashed
            features.
    """
    pointArr = point.split(',')
    label = float(pointArr[0])
    data = pointArr[1:]
    return LabeledPoint(label, SparseVector(numBuckets, hashFunction(numBuckets, zip(range(len(data)), data))))

def computeLogLoss(p, y):
    """Calculates the value of log loss for a given probabilty and label.

    Note:
        log(0) is undefined, so when p is 0 we need to add a small value (epsilon) to it
        and when p is 1 we need to subtract a small value (epsilon) from it.

    Args:
        p (float): A probabilty between 0 and 1.
        y (int): A label.  Takes on the values 0 and 1.

    Returns:
        float: The log loss value.
    """
    epsilon = 10e-12
    
    if p == 0:
        p += epsilon
    elif p == 1:
        p -= epsilon
        
    if y == 1:
        return -1.0*log(p)
    else:
        return -1.0*log(1-p)
    
def getP(x, w, intercept):
    """Calculate the probability for an observation given a set of weights and intercept.

    Note:
        We'll bound our raw prediction between 20 and -20 for numerical purposes.

    Args:
        x (SparseVector): A vector with values of 1.0 for features that exist in this
            observation and 0.0 otherwise.
        w (DenseVector): A vector of weights (betas) for the model.
        intercept (float): The model's intercept.

    Returns:
        float: A probability between 0 and 1.
    """
    rawPrediction = x.dot(np.array(w)) + intercept

    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    return 1.0/(1.0 + exp(-1.0*rawPrediction))

def evaluateResults(model, data):
    """Calculates the log loss for the data given the model.

    Args:
        model (LogisticRegressionModel): A trained logistic regression model.
        data (RDD of LabeledPoint): Labels and features for each observation.

    Returns:
        float: Log loss for the data.
    """
    labelAndPreds = data.map(lambda lp: (lp.label, getP(lp.features, model.weights, model.intercept)))
    return labelAndPreds.map(lambda lp: computeLogLoss(lp[1], lp[0])).reduce(lambda a,b:a+b)/labelAndPreds.count()

if __name__ == "__main__":
	sc = SparkContext(appName="wikiPageRank")

	#train_data_folder = './dac_sample.txt'
	#val_data_folder = './dac_sample.txt'
	#test_data_folder = './dac_sample.txt'

	numIters = 50
	stepSize = 10.
	regParam = 1e-6
	#regType = 'l2'
	regType = None
	includeIntercept = True
	numBucketsCTR = 1000

	train_data_folder = 's3n://criteo-dataset/rawdata/train/'
	val_data_folder = 's3n://criteo-dataset/rawdata/validation/'
	test_data_folder = 's3n://criteo-dataset/rawdata/test/'
	
	output_loc = 's3://kuanlin.aws.dev/criteo_baseline/'

	train_data_raw = sc.textFile(train_data_folder).map(lambda x: x.replace('\t', ','))
	hashTrainData = train_data_raw.map(lambda line: parseHashPoint(line, numBucketsCTR))

	val_data_raw = sc.textFile(val_data_folder).map(lambda x: x.replace('\t', ','))
	hashValData = val_data_raw.map(lambda line: parseHashPoint(line, numBucketsCTR))

	test_data_raw = sc.textFile(test_data_folder).map(lambda x: x.replace('\t', ','))
	hashTestData = test_data_raw.map(lambda line: parseHashPoint(line, numBucketsCTR))

	model_logistic = LogisticRegressionWithSGD.train(hashTrainData, iterations=numIters, step=stepSize,
											 regParam=regParam, regType=regType, intercept=includeIntercept)

	train_logloss = evaluateResults(model_logistic, hashTrainData)
	validation_logloss = evaluateResults(model_logistic, hashValData)
	test_logloss = evaluateResults(model_logistic, hashTestData)

	print "Train Logloss: %s"%train_logloss
	print "Validation Logloss: %s"%validation_logloss
	print "Test Logloss: %s"%test_logloss

	trainPredictionAndLabels = hashTrainData.map(lambda lp: (float(model_logistic.predict(lp.features)), lp.label))
	trainMetrics = BinaryClassificationMetrics(trainPredictionAndLabels)
	print("Train Area under ROC = %s" % trainMetrics.areaUnderROC)

	valPredictionAndLabels = hashValData.map(lambda lp: (float(model_logistic.predict(lp.features)), lp.label))
	valMetrics = BinaryClassificationMetrics(valPredictionAndLabels)
	print("Train Area under ROC = %s" % valMetrics.areaUnderROC)

	testPredictionAndLabels = hashTrainData.map(lambda lp: (float(model_logistic.predict(lp.features)), lp.label))
	testMetrics = BinaryClassificationMetrics(testPredictionAndLabels)
	print("Train Area under ROC = %s" % testMetrics.areaUnderROC)
	
	sc.stop()
	
	output_file = 'criteo_baseline.txt'
	writer = open(output_file, 'w')
	writer.write("Train Logloss: %s\n"%train_logloss)
	writer.write("Validation Logloss: %s\n"%validation_logloss)
	writer.write("Test Logloss: %s\n"%test_logloss)
	writer.write("Train Area under ROC = %s\n" % trainMetrics.areaUnderROC)
	writer.write("Train Area under ROC = %s\n" % valMetrics.areaUnderROC)
	writer.write("Train Area under ROC = %s\n" % testMetrics.areaUnderROC)
	writer.close()
	os.system("aws s3 cp %s %s"%(output_file, output_loc + output_file))
	