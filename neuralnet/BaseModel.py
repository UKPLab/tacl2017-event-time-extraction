import random
import time
import numpy as np
from itertools import chain,combinations
import json

class BaseModel(object):
    def __init__(self, name):
        self.best_model_acc = 0
        self.name = name
    
    def getPrecision(self, pred_test, yTest):
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
    
    
    
    
    def getAccuracy(self, model, input, labels):
        pred_test = model.predict_classes(input, verbose=False)  
        
        acc =  np.sum(pred_test == labels) / float(len(labels))
        
    
        prec = self.getPrecision(pred_test, labels)
        rec = self.getPrecision(labels, pred_test)
        f1 = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
        
    
        return acc, prec, rec, f1
    
    
    def optimizeModel(self, max_evals):
        print self.name
        
        best_model_acc = self.best_model_acc
        
        for model_nr in xrange(max_evals):
            params = {}
            for key, values in self.space.iteritems():
                params[key] = random.choice(values)
          
            print "Model nr. ", model_nr
            max_acc, best_model_acc = self.run_model(params, best_model_acc)
            self.best_model_acc = best_model_acc
            print "Max acc: %.4f" % (max_acc)
            
            
    def powerset(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return list(chain.from_iterable(combinations(s, r) for r in range(0, len(s)+1)))
    
    def save_params(self, params,  featureSet, modelOutputPathName):
        fOut = open(modelOutputPathName+".params.txt", 'w')
        fOut.write(str(params))
        fOut.close()
        
        fOut = open(modelOutputPathName+".featureSet.txt", 'w')
        fOut.write(str(featureSet))
        fOut.close()