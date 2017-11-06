from neuralnet.TwoTokenClassifier import TwoTokenClassifier

name = '2_SingleDay/2_TimeRelevant'


classifier = TwoTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

