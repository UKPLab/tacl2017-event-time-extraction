import theano
import numpy as np
import random
import cPickle as pkl
from keras.models import load_model
import os
from keras.utils import np_utils


def getPrecision(pred_test, yTest):
    #Precision for non-vague
    non_vague = 0
    correct_non_vague = 0
   
    for idx in xrange(len(pred_test)):
        if pred_test[idx] != 0:
            non_vague += 1
           
            if pred_test[idx] == yTest[idx]:
                correct_non_vague += 1
   
    if non_vague == 0:
        return 0
   
    return correct_non_vague / float(non_vague)
    
def getAccuracy(model, input, labels):
    pred_test = model.predict_classes(input, verbose=False)  
    
    acc =  np.sum(pred_test == labels) / float(len(labels))
  

    prec = getPrecision(pred_test, labels)
    rec = getPrecision(labels, pred_test)
    f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)

    return acc, prec, rec, f1, pred_test

def getFeatureSet(modelName):
    fIn = open(modelName+".featureSet.txt")
    featureSet = eval(fIn.readline().strip())
    fIn.close()
    return featureSet

def SingleTokenOutput(name, labelsMapping):
    f = open('pkl/'+name+'/data.pkl', 'rb')    
    trainSet  = pkl.load(f)
    devSet = pkl.load(f)
    testSet = pkl.load(f)
    fullTestSet = pkl.load(f)
    f.close()
    
    modelDir = 'models/'+name
    modelNames = []
    
    for file in os.listdir( modelDir):
        if file.endswith(".h5") and file.startswith('0.'):
            modelNames.append(modelDir+'/'+file)
            
    modelNames.sort()
    
    maxModelName = modelNames[-1]
    
    
    featureSet = getFeatureSet(maxModelName)    
    model = load_model(maxModelName)
    
    acc, prec, rec, f1, pred_test = getAccuracy(model, [testSet[ft] for ft in featureSet], testSet['labels'])
    
    print name
    print maxModelName
    print "Acc: %.4f | F1: %.4f on test\n" % (acc, f1)

    inv_map = {v: k for k, v in labelsMapping.items()}
    
    outputPath = 'output/'+name
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    fOut = open(outputPath+'/output.txt', "w")
    
    pred_full_test = model.predict_classes([fullTestSet[ft] for ft in featureSet], verbose=False)  
    
    for idx in xrange(len(pred_full_test)):
        eId = fullTestSet['features'][idx]['eId']
        corpusFile = fullTestSet['features'][idx]['corpusFile'] 
        goldLabel = fullTestSet['features'][idx]['label'] 
        predLabel = inv_map[pred_full_test[idx]]
        fOut.write("%s\t%s\t%s\t%s\n" % (corpusFile, eId, goldLabel, predLabel))
    

    
def TwoTokenOutput(name, labelsMapping):
    f = open('pkl/'+name+'/data.pkl', 'rb')    
    trainSet  = pkl.load(f)
    devSet = pkl.load(f)
    testSet = pkl.load(f)
    fullTestSet = pkl.load(f)
    f.close()
    
    modelDir = 'models/'+name
    modelNames = []
    
    for file in os.listdir( modelDir):
        if file.endswith(".h5") and file.startswith('0.'):
            modelNames.append(modelDir+'/'+file)
            
    modelNames.sort()
    
    maxModelName = modelNames[-1]
    
    
    featureSet = getFeatureSet(maxModelName)   
    model = load_model(maxModelName)
    
    acc, prec, rec, f1, pred_test = getAccuracy(model, [testSet[ft] for ft in featureSet], testSet['labels'])
    
    print name
    print maxModelName
    print "Acc: %.4f | F1: %.4f on test\n" % (acc, f1)

    inv_map = {v: k for k, v in labelsMapping.items()}
    
    outputPath = 'output/'+name
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    fOut = open(outputPath+'/output.txt', "w")
    
    pred_full_test = model.predict_classes([fullTestSet[ft] for ft in featureSet], verbose=False)  
    
    for idx in xrange(len(pred_full_test)):
        eId = fullTestSet['features'][idx]['eId']
        tId = fullTestSet['features'][idx]['tId']
        corpusFile = fullTestSet['features'][idx]['corpusFile'] 
        goldLabel = fullTestSet['features'][idx]['label'] 
        predLabel = inv_map[pred_full_test[idx]]
        fOut.write("%s\t%s\t%s\t%s\t%s\n" % (corpusFile, eId, tId, goldLabel, predLabel))
    
    
SingleTokenOutput('1_EventType', {'singleday':0, 'multiday':1})
SingleTokenOutput('2_SingleDay/1_DCTRelations', {'afterdct': 0, 'beforedct': 1,'dct': 2})
TwoTokenOutput('2_SingleDay/2_TimeRelevant', {'nonrelevant':0, 'relevant':1})
TwoTokenOutput('2_SingleDay/3_TimexRelations', {'a': 0, 'b': 1, 's': 2})

SingleTokenOutput('3_MultiDay/1_DCTRelations', {'afterdct': 0, 'beforedct': 1,'dct': 2})
TwoTokenOutput('3_MultiDay/2_Begin_TimeIsRelevant', {'nonrelevant':0, 'relevant':1})
TwoTokenOutput('3_MultiDay/3_Begin_TimexRelations', {'a': 0, 'b': 1, 's': 2})

TwoTokenOutput('3_MultiDay/4_End_TimeIsRelevant', {'nonrelevant':0, 'relevant':1})
TwoTokenOutput('3_MultiDay/5_End_TimexRelations', {'a': 0, 'b': 1, 's': 2})



