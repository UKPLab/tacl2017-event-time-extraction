from neuralnet.TwoTokenClassifier import TwoTokenClassifier

name = '3_MultiDay/3_Begin_TimexRelations'


classifier = TwoTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

