from neuralnet.TwoTokenClassifier import TwoTokenClassifier

name = '2_SingleDay/3_TimexRelations'


classifier = TwoTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

