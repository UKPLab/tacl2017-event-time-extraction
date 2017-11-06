from neuralnet.SingleTokenClassifier import SingleTokenClassifier

name = '2_SingleDay/1_DCTRelations'


classifier = SingleTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

