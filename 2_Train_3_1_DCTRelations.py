from neuralnet.SingleTokenClassifier import SingleTokenClassifier

name = '3_MultiDay/1_DCTRelations'


classifier = SingleTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

