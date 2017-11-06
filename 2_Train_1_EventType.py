from neuralnet.SingleTokenClassifier import SingleTokenClassifier

name = '1_EventType'


classifier = SingleTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

