# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20,50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    
    #start from using only one k
    
    features=self.features
    print 'length of features' ,len(features) 
    print 'length of trainingData' ,len(trainingData) 
    # 100 training labels & data, and each element of trainingData contains the binary info of 784 points, or to say, features
    # legalLabels are all the labels that need classification
    legalLabels = self.legalLabels 
    
   
    
    n = len(trainingData)
    num= util.Counter()
    p= util.Counter()
    for y in trainingLabels :
        num[y]+=1
    for y in legalLabels :
        p[y]= num[y]/ float(n)
    # now p[y] contains p[y] we need
    #print p    
    # we still use num & p, just change the keys
    for trainingDatum in trainingData:
        index= trainingData.index(trainingDatum)
        y = trainingLabels[index]
        for (i,j) in trainingDatum.keys():
            if trainingDatum[(i,j)]==1:
                num[(y,(i,j))]+=1
    #print len(num), as it should be, 7850
    # we now have the data p[y] & p[(y,(i,j))]
    # from now on we need k to smooth p[(y,(i,j))]
    grade = util.Counter() # to measure which k is the best
    for k in kgrid:
        print 'tuning' ,k         
        for (i,j) in features:
            for y in legalLabels:
                p[(y,(i,j))] = (num[(y,(i,j))]+k) / float(num[y]+2*k)
        self.p = p
        guesses = self.classify(validationData)
        correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        grade[k] = correct
    print 'The correctness of each auto tuning variable among 100', grade
    #find the best k
    k = grade.argMax() 
    # satisfy the requirement of: In case of ties, prefer the lowest value of k.
    print 'The best k isssssssssssssssss!!!', k       
    for (i,j) in features:
            for y in legalLabels:
                p[(y,(i,j))] = (num[(y,(i,j))]+k) / float(num[y]+2*k)
    self.p = p  
    
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    features = self.features
    legalLabels = self.legalLabels
    logJoint = util.Counter()
    p=self.p
    for y in legalLabels:
        logJoint[y]+=math.log(p[y])
        for (i,j) in features:
            if datum[i,j]==1:
                logJoint[y]+=math.log(p[(y,(i,j))])
            else:
                logJoint[y]+=math.log(1-p[(y,(i,j))])
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    legalLabels = self.legalLabels
    features = self.features
    featuresOdds = []
    p=self.p
    values = util.PriorityQueue()
    for (i,j) in features:
        # reverse, since if we want p1 /p2 to be biggest, then p2/ p1 to be the smallest. 
        #Because the queue pop the lowest
        priority = p[(label2,(i,j))]/ p[(label1,(i,j))] 
        values.push((i,j),priority)
    for i in range(100):
        featuresOdds+=[values.pop()]
    print featuresOdds
    return featuresOdds
    # the answer should be 'a'
    

    
      
