import numpy as np
import cPickle as pkl
import gzip
import os




def createEmbeddingsFile():
    folder = 'input/'
    
    embeddingsPath = '0_Preprocessing/embeddings/levy_dependency_based.words.vocab.gz'
    words = {}
    
    filePaths = []
    
    for root, subdirs, files in os.walk(folder):
        for fName in files:
            if fName.endswith('.txt'):
                filePaths.append(root+'/'+fName)
       


    for fName in filePaths:
        for line in open(fName):
            splits = eval(line.strip())
            
            event = splits["Token[0]"]
            words[event.lower()] = True
            
            if 'TimeTokenFirst' in splits:
                time = splits['TimeTokenFirst']
                words[time.lower()] = True
            
            if 'textInBetween' in splits:
                tokens = splits["textInBetween"]   
                for token in tokens:
                    words[token.lower()] = True
                    
            if 'sentence' in splits:
                tokens = splits["sentence"]   
                for token in tokens:
                    words[token.lower()] = True
            
    # :: Read in word embeddings ::
    
    word2Idx = {}
    embeddings = []
    
    if embeddingsPath.endswith(".gz"):
        embeddingsIn = gzip.open(embeddingsPath, 'rb')
    else:
        embeddingsIn = open(embeddingsPath)
    
    for line in embeddingsIn:
        split = line.strip().split(" ")
        word = split[0]
        
        if len(word2Idx) == 0: #Add padding+unknown
            word2Idx["PADDING"] = len(word2Idx)
            vector = np.zeros(len(split)-1) #2*0.1*np.random.rand(len(split)-1)-0.1
            embeddings.append(vector)
            
            word2Idx["UNKNOWN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split)-1) #2*0.1*np.random.rand(len(split)-1)-0.1
            embeddings.append(vector)
    
        if word.lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            embeddings.append(vector)
            word2Idx[word] = len(word2Idx)
            
    embeddings = np.array(embeddings)
    
    print "Embeddings shape: ", embeddings.shape
    print "Len words: ", len(words)
    
    f = open('pkl/embeddings.pkl', 'wb')
    pkl.dump(embeddings, f, -1)
    pkl.dump(word2Idx, f, -1)
    f.close()

if os.path.exists('pkl/embeddings.pkl'):
    inp = raw_input("Overwrite embeddings.pkl (y/n): ")
    if inp.strip() == 'y':
        createEmbeddingsFile() 
    else:
        print "Skipt embeddings.pkl"
else:
    createEmbeddingsFile()
    
    
########################################
#
# Create task specific pickle files
#
########################################





def createMatrices(file, word2Idx, maxSentenceLen, extendMapping, labelsMapping, aspectMapping, typeMapping, tenseMapping, eventClassMapping, distanceMapping, sentenceLengthMapping):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    eventMatrix = []
    timeMatrix = []

    sentenceLengths = []
    sentenceMatrix = []
    positionMatrix_e = []
    positionMatrix_t = []

    aspectMatrix = []
    typeMatrix = []
    tenseMatrix = []
    eventClassMatrix = []
    featuresList = []
    
    
    minDistance = 0
    for distanceKey in distanceMapping.iterkeys():
        if isinstance(distanceKey, (int, long)):
            minDistance = min(minDistance, int(distanceKey))
    
    for line in open(file):
        features = eval(line.strip())
        featuresList.append(features)
        label = features['label']
        event = features["Token[0]"]
        eventPosition = int(features["eventPosition"])
        tokens = features["textInBetween"] if "textInBetween" in features else features["sentence"]
        
        aspect = features["aspect"]
        tense = features["tense"]
        eventClass = features["eventClass"]
        
        labels.append(labelsMapping[label.lower()] if label.lower() in labelsMapping else -1)
        eventMatrix.append(getWordIdx(event, word2Idx))
        
        aspectMatrix.append(getMappingIdx(aspect, aspectMapping, extendMapping))
        tenseMatrix.append(getMappingIdx(tense, tenseMapping, extendMapping))
        eventClassMatrix.append(getMappingIdx(eventClass, eventClassMapping, extendMapping))
        
        if 'TimeTokenFirst' in features:
            time = features['TimeTokenFirst']
            timeMatrix.append(getWordIdx(time, word2Idx))
        
        if 'type' in features:
            type = features["type"]
            typeMatrix.append(getMappingIdx(type, typeMapping, extendMapping))
       
        timePosition =  int(features["timeFirstPosition"])  if 'timeFirstPosition' in features else 0
            

        if len(tokens) in sentenceLengthMapping:
            sentenceLengths.append(sentenceLengthMapping[len(tokens)])
        else:
            sentenceLengths.append(sentenceLengthMapping['GreaterMax'])

        
        tokenIds = np.zeros(maxSentenceLen)
        positionValues_e = np.zeros(maxSentenceLen)
        positionValues_t = np.zeros(maxSentenceLen)
        for idx in xrange(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)
            
            distance_e = idx - eventPosition
            distance_t = idx - timePosition

            if distance_e in distanceMapping:
                positionValues_e[idx] = distanceMapping[distance_e]
            elif distance_e <= minDistance:
                positionValues_e[idx] = distanceMapping['LowerMin']
            else:
                positionValues_e[idx] = distanceMapping['GreaterMax']
            
            if distance_t in distanceMapping:
                positionValues_t[idx] = distanceMapping[distance_t]
            elif distance_t <= minDistance:
                positionValues_t[idx] = distanceMapping['LowerMin']
            else:
                positionValues_t[idx] = distanceMapping['GreaterMax']

        sentenceMatrix.append(tokenIds)
        positionMatrix_e.append(positionValues_e)
        positionMatrix_t.append(positionValues_t)

        
    labels = np.array(labels, dtype='int32')
    eventMatrix = np.expand_dims(np.array(eventMatrix, dtype='int32'), axis=1)
    timeMatrix = np.expand_dims(np.array(timeMatrix, dtype='int32'), axis=1)
    aspectMatrix = np.expand_dims(np.array(aspectMatrix, dtype='int32'), axis=1)
    typeMatrix = np.expand_dims(np.array(typeMatrix, dtype='int32'), axis=1)
    tenseMatrix = np.expand_dims(np.array(tenseMatrix, dtype='int32'), axis=1)
    eventClassMatrix = np.expand_dims(np.array(eventClassMatrix, dtype='int32'), axis=1)
    sentenceLengths = np.expand_dims(np.array(sentenceLengths, dtype='int32'), axis=1)
    
    
    sentenceMatrix = np.array(sentenceMatrix, dtype='int32')
    positionMatrix_e = np.array(positionMatrix_e, dtype='int32')
    positionMatrix_t = np.array(positionMatrix_t, dtype='int32')
    


    return {'labels': labels,
            'event':eventMatrix, 
            'time':timeMatrix, 
            'sentence':sentenceMatrix, 
            'positions_e':positionMatrix_e, 
            'positions_t':positionMatrix_t, 
            'aspect':aspectMatrix, 
            'tense':tenseMatrix, 
            'eventClass':eventClassMatrix, 
            'type': typeMatrix,
            'sentence_len': sentenceLengths,
            'features': featuresList}
    

def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN"]


def getValue(splits, featureName):
    """From a crfsuite feature file, returns the feature with the name featureName """
    for split in splits:
        if split.startswith(featureName):
            return split[split.find("=")+1:]
    
    return None


def getMappingIdx(value, mapping, extendMapping):
    if value in mapping:
        return mapping[value]
    
    if extendMapping:
        if 'UNKNOWN' not in mapping:
            mapping['UNKNOWN'] = len(mapping)
            
        mapping[value] = len(mapping)
        return mapping[value]
    else:        
        return mapping['UNKNOWN']


def createPickleFiles(name, labelsMapping):
    folder = 'input/'+name
    files = [folder+'/train.txt', folder+'/dev.txt', folder+'/test.txt', folder+'/full_test.txt']
    
    outputFilePath = 'pkl/'+name
    if not os.path.exists(outputFilePath):
        os.makedirs(outputFilePath)
        
    outputFilePath += '/data.pkl'
    
    if os.path.exists(outputFilePath):
        inp = raw_input("Overwrite "+outputFilePath+" (y/n): ")
        if inp.strip() != 'y':
            print "Skip "+outputFilePath
            return
        
    maxSentenceLen = 0
    for fName in files:
        for line in open(fName):
            splits = eval(line.strip())
            tokens = splits["textInBetween"] if 'textInBetween' in splits else splits['sentence']
            maxSentenceLen = max(maxSentenceLen, len(tokens))

    
    
    f = open('pkl/embeddings.pkl', 'rb')
    embeddings = pkl.load(f)
    word2Idx = pkl.load(f)
    f.close()
    
    
    
    aspectMapping = {}
    typeMapping = {}
    tenseMapping = {}
    eventClassMapping = {}
    
    distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}
    minDistance = -30
    maxDistance = 30
    for dis in xrange(minDistance,maxDistance+1):
        distanceMapping[dis] = len(distanceMapping)
        
    sentenceLengthMapping = {'PADDING': 0, 'GreaterMax': 1}
    minDistance = 0
    maxDistance = 50
    for dis in xrange(minDistance,maxDistance+1):
        sentenceLengthMapping[dis] = len(sentenceLengthMapping)
    
    # :: Create token matrix ::
    train_set = createMatrices(files[0], word2Idx, maxSentenceLen, True, labelsMapping, aspectMapping, typeMapping, tenseMapping, eventClassMapping, distanceMapping, sentenceLengthMapping)
    dev_set = createMatrices(files[1], word2Idx, maxSentenceLen, False, labelsMapping, aspectMapping, typeMapping, tenseMapping, eventClassMapping, distanceMapping, sentenceLengthMapping)
    test_set = createMatrices(files[2], word2Idx, maxSentenceLen, False, labelsMapping, aspectMapping, typeMapping, tenseMapping, eventClassMapping, distanceMapping, sentenceLengthMapping)
    full_test_set = createMatrices(files[3], word2Idx, maxSentenceLen, False, labelsMapping, aspectMapping, typeMapping, tenseMapping, eventClassMapping, distanceMapping, sentenceLengthMapping)
    
    f = open(outputFilePath, 'wb')
    pkl.dump(train_set, f, -1)
    pkl.dump(dev_set, f, -1)
    pkl.dump(test_set, f, -1)
    pkl.dump(full_test_set, f, -1)
    f.close()
    
    print "\n\nData stored at "+outputFilePath
    
    
    print 'train_set:'
    labelDist = {}
    for label in train_set['labels']:
        if label not in labelDist:
            labelDist[label] = 0
        labelDist[label] += 1
    for label, cnt in labelDist.iteritems():
        print "%s: %d" % (label, cnt)
    
    
    
    print 'dev_set:'
    labelDist = {}
    for label in dev_set['labels']:
        if label not in labelDist:
            labelDist[label] = 0
        labelDist[label] += 1
    for label, cnt in labelDist.iteritems():
        print "%s: %d" % (label, cnt)
    
    
    print 'test_set:'
    labelDist = {}
    for label in test_set['labels']:
        if label not in labelDist:
            labelDist[label] = 0
        labelDist[label] += 1
    for label, cnt in labelDist.iteritems():
        print "%s: %d" % (label, cnt)
    
    print "total: %d" % len(train_set['labels'])
    
    
createPickleFiles('1_EventType', {'singleday':0, 'multiday':1})  

createPickleFiles('2_SingleDay/1_DCTRelations', {'afterdct': 0, 'beforedct': 1,'dct': 2})    
createPickleFiles('2_SingleDay/2_TimeRelevant', {'nonrelevant':0, 'relevant':1})
createPickleFiles('2_SingleDay/3_TimexRelations', {'a': 0, 'b': 1, 's': 2})



createPickleFiles('3_MultiDay/1_DCTRelations', {'afterdct': 0, 'beforedct': 1, 'dct': 2, 'other': 3})   

createPickleFiles('3_MultiDay/2_Begin_TimeIsRelevant', {'nonrelevant':0, 'relevant':1})   
createPickleFiles('3_MultiDay/3_Begin_TimexRelations', {'a': 0, 'b': 1, 's': 2})  

createPickleFiles('3_MultiDay/4_End_TimeIsRelevant', {'nonrelevant':0, 'relevant':1})   
createPickleFiles('3_MultiDay/5_End_TimexRelations', {'a': 0, 'b': 1, 's': 2})  

print "--DONE---"