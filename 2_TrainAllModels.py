from neuralnet.SingleTokenClassifier import SingleTokenClassifier
from neuralnet.TwoTokenClassifier import TwoTokenClassifier
from random import shuffle

classifiers = [
          SingleTokenClassifier('1_EventType'),
          SingleTokenClassifier('2_SingleDay/1_DCTRelations'),
          TwoTokenClassifier('2_SingleDay/2_TimeRelevant'),
          TwoTokenClassifier('2_SingleDay/3_TimexRelations'),
          SingleTokenClassifier('3_MultiDay/1_DCTRelations'),
          TwoTokenClassifier('3_MultiDay/2_Begin_TimeIsRelevant'),
          TwoTokenClassifier('3_MultiDay/3_Begin_TimexRelations'),
          TwoTokenClassifier('3_MultiDay/4_End_TimeIsRelevant'),
          TwoTokenClassifier('3_MultiDay/5_End_TimexRelations')
          ]

shuffle(classifiers)

for i in xrange(1000):
    print "\n\n\n--------------------------"
    print "Iteration: ", i
    print "--------------------------\n\n\n"
    for classifier in classifiers:
        classifier.optimizeModel(1)


print "DONE"

