from neuralnet.TwoTokenClassifier import TwoTokenClassifier

name = '3_MultiDay/2_Begin_TimeIsRelevant'


classifier = TwoTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

