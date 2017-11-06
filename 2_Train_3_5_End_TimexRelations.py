from neuralnet.TwoTokenClassifier import TwoTokenClassifier

name = '3_MultiDay/5_End_TimexRelations'


classifier = TwoTokenClassifier(name)
classifier.optimizeModel(1000)

print "DONE"

