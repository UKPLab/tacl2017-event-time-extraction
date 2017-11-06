from neuralnet.TwoTokenClassifier import TwoTokenClassifier

name = '3_MultiDay/4_End_TimeIsRelevant'


classifier = TwoTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

