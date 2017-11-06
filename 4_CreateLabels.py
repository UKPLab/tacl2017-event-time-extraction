import re
import nltk

from helpfunctions import mutual_exclusive_score

eventTimePath = '0_Preprocessing/corpus/eventtime/event-times_normalized.tab'
eventTimes = {}
predLabelPosition = -1

def getDates(value):
    """
    Normalizes TimeX Value, Normalize seasons and weeks
    return the normalized dates in value
    """
    dates = {}
    before = ''
    after = ''

    #normalize season short forms
    for match in re.finditer("[0-9]{4}-FA", value):
        before = match.group()[:5] + "12-21"
        after = match.group()[:5] + "09-23"

    for match in re.finditer("[0-9]{4}-SP", value):
        before = match.group()[:5] + "06-20"
        after = match.group()[:5] + "03-20"

    for match in re.finditer("[0-9]{4}-WI", value):
        before = match.group()[:5] + "03-19"
        after = match.group()[:5] + "12-22"

    for match in re.finditer("[0-9]{4}-SU", value):
        before = match.group()[:5] + "09-22"
        after = match.group()[:5] + "06-21"

    #normalize '1997-12'
    for match in re.finditer("[0-9]{4}-[0-9]{2}$", value):
        # print 'value : ' , value
        month = match.group()[5:7]
        
        daysInMonth = [0,
                       31, #Jan
                       28, #Feb
                       31, #Mar
                       30, #Apr
                       31, #May
                       30, #Jun
                       31, #Jul
                       31, #Aug
                       30, #Sep
                       31, #Okt
                       30, #Nov
                       31] #Dez
        before = match.group()[:7] + "-" + str(daysInMonth[int(month)])
        after = match.group()[:7] + "-01"
        

    #normalize '1997'
    for match in re.finditer("[0-9]{4}$", value):
        # print 'value : ' , value
        before = match.group()[:4] + "-12-31"
        after = match.group()[:4] + "-01-01"

    #normalize week numbers (e.g. 1988-W23)
    if re.match('[0-9]{4}-W[0-9]{2}', value):
        year = int(value[0:4])
        week = int(value[6:8])
        sunday = Week(year, week).sunday()
        nextSaturday = Week(year, week).saturday() + timedelta(days=7)

        after = sunday.strftime("%Y-%m-%d")
        before = nextSaturday.strftime("%Y-%m-%d")

    #normalize quarter short forms
    for match in re.finditer("[0-9]{4}-Q1", value):
        after = match.group()[:5] + "01-01"
        before = match.group()[:5] + "03-31"

    for match in re.finditer("[0-9]{4}-Q2", value):
        after = match.group()[:5] + "04-01"
        before = match.group()[:5] + "09-30"

    for match in re.finditer("[0-9]{4}-Q3", value):
        after = match.group()[:5] + "07-01"
        before = match.group()[:5] + "09-30"

    for match in re.finditer("[0-9]{4}-Q4", value):
        after = match.group()[:5] + "10-01"
        before = match.group()[:5] + "12-31"

    if before != '' and after != '':
        dates['before'] = before
        dates['after'] = after
        return dates

    
    if re.match('[0-9]{4}-[0-9]{2}-[0-9]{2}T', value):
        value = value[:10]

    dates['sim'] = value
    return dates



def narrowDown(event, prefix, candidatesBefore, candidatesAfter, majorityVote=True):
    if prefix+"_TimeRelevant" in event:
        for tId in event[prefix+'_TimeRelevant']:
            if event[prefix+'_TimeRelevant'][tId] == 'b':                                
                timexValue = timexValues[corpusFile][tId]
                
                dates = getDates(timexValue)
                
                if 'before' in dates:
                    candidatesBefore.append(dates['before'])
                    
                #if 'after' in dates:
                #    candidatesAfter.append(dates['after'])
                
                if 'sim' in dates:
                    candidatesBefore.append(dates['sim'])                    
               
            elif event[prefix+'_TimeRelevant'][tId] == 'a':
                timexValue = timexValues[corpusFile][tId]
                
                dates = getDates(timexValue)
                
                #if 'before' in dates:
                #    candidatesBefore.append(dates['before'])
                
                if 'after' in dates:
                    candidatesAfter.append(dates['after'])
                    
                
                if 'sim' in dates:
                    candidatesAfter.append(dates['sim'])             
                    
                
    
    candidatesBefore.sort()
    candidatesAfter.sort()
    
    afterValue = None
    beforeValue = None
    
    if majorityVote:
        freqCandidatesBefore = nltk.FreqDist()
        for candidateBefore in candidatesBefore:
            freqCandidatesBefore[candidateBefore] += 1
           
        if len(freqCandidatesBefore) > 1: 
            mostCommon, secondMostCommon =  freqCandidatesBefore.most_common(2)
            
            if mostCommon[1] > secondMostCommon[1]:
                beforeValue = mostCommon[0]
                
        freqCandidatesAfter = nltk.FreqDist()
        for candidateAfter in candidatesAfter:
            freqCandidatesAfter[candidateAfter] += 1
           
        if len(freqCandidatesAfter) > 1: 
            mostCommon, secondMostCommon =  freqCandidatesAfter.most_common(2)
            
            if mostCommon[1] > secondMostCommon[1]:
                afterValue = mostCommon[0]
            
        
        
    if afterValue == None and len(candidatesAfter) > 0:
        afterValue = candidatesAfter[-1]
    if beforeValue == None and len(candidatesBefore) > 0:
        beforeValue = candidatesBefore[0]
        
    predLabel = ""
    if afterValue == beforeValue and afterValue != None:
        predLabel = afterValue
    else:
        if afterValue != None:
            predLabel += "after"+afterValue
        if beforeValue != None:
            predLabel += "before"+beforeValue
    
   
    return predLabel


# :: Read Event Times ::


for line in open(eventTimePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[4]
    eventTime = splits[-1]
    
    if corpusFile not in eventTimes:
        eventTimes[corpusFile] = {}
        
    eventTimes[corpusFile][eId] = eventTime


# :: Read DCT values ::
dctPath = '0_Preprocessing/corpus/dct.txt'
dctValues = {}

for line in open(dctPath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    dctValue = splits[1][0:10]
    
            
    dctValues[corpusFile] = dctValue


# :: Read Timex values ::
timexPath = '0_Preprocessing/corpus/timex-attributes.tab'
timexValues = {}

for line in open(timexPath):
    splits = line.strip().split('\t')
    
    if splits[6] == 'value':
        corpusFile = splits[0]
        tId = splits[4]
        timeValue = splits[-1]
        
        if timeValue in ['PRESENT_REF', 'FUTURE_REF', 'PAST_REF']:
            timeValue = dctValues[corpusFile]
            
        
            
        if len(timeValue) > 10:
            timeValue = timeValue[0:10]
        
        if corpusFile not in timexValues:
            timexValues[corpusFile] = {}
            
        timexValues[corpusFile][tId] = timeValue
        

########################################################################
########################################################################
########################################################################
########################################################################

# :: Read target events ::
fullTestSet = 'input/1_EventType/full_test.txt'
events = {}

for line in open(fullTestSet):
    splits = eval(line.strip())
    
    corpusFile = splits['corpusFile']
    eId = splits['eId']
    
    if corpusFile not in events:
        events[corpusFile] = {}
        
    events[corpusFile][eId] = {}

    
# :: Read Event Type ::
eventTypePath = 'output/1_EventType/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    predLabel = splits[predLabelPosition].lower()
    
    events[corpusFile][eId]["EventType"] = predLabel
    
    
########## Single Day Events ##########

# :: Read SingleDay DCT ::
eventTypePath = 'output/2_SingleDay/1_DCTRelations/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    predLabel = splits[predLabelPosition].lower()
    
    events[corpusFile][eId]["SingleDay_DCTRelations"] = predLabel
    
# :: Read SingleDay Relevant TimeValues ::
eventTypePath = 'output/2_SingleDay/2_TimeRelevant/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    tId = splits[2]
    predLabel = splits[predLabelPosition].lower()
    
    if predLabel == 'relevant':    
        if 'SingleDay_TimeRelevant' not in events[corpusFile][eId]:
            events[corpusFile][eId]["SingleDay_TimeRelevant"] = {}
            
        events[corpusFile][eId]["SingleDay_TimeRelevant"][tId] = None
        
# :: Read SingleDay TimexRelations ::        
eventTypePath = 'output/2_SingleDay/3_TimexRelations/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    tId = splits[2]
    predLabel = splits[predLabelPosition].lower()
    
    
    if "SingleDay_TimeRelevant" in events[corpusFile][eId] and tId in events[corpusFile][eId]["SingleDay_TimeRelevant"]:        
        events[corpusFile][eId]["SingleDay_TimeRelevant"][tId] = predLabel
    
########## Multi Day Events ##########    

# :: Read Multi Day DCT ::
numLabels = 0
numCorrectLabels = 0
freqDist = {}

eventTypePath = 'output/3_MultiDay/1_DCTRelations/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    predLabel = splits[predLabelPosition].lower()
    correctLabel = splits[-2].lower()
    
    events[corpusFile][eId]["MultiDay_DCTRelations"] = predLabel
    
    if correctLabel != 'n/a':
        numLabels += 1
        numCorrectLabels += 1 if correctLabel == predLabel else 0
        
        if predLabel not in freqDist:
            freqDist[predLabel] = {}
        if correctLabel not in freqDist[predLabel]:
            freqDist[predLabel][correctLabel] = 0
            
        freqDist[predLabel][correctLabel] += 1

print freqDist
print "Accuracy for "+eventTypePath+": ", float(numCorrectLabels) / numLabels    
    
# :: Read MultiDay Begin Relevant TimeValues ::
numLabels = 0
numCorrectLabels = 0
freqDist = {}

eventTypePath = 'output/3_MultiDay/2_Begin_TimeIsRelevant/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    tId = splits[2]
    predLabel = splits[predLabelPosition].lower()
    correctLabel = splits[-2].lower()

    if predLabel == 'relevant':    
        if 'MultiDay_Begin_TimeRelevant' not in events[corpusFile][eId]:
            events[corpusFile][eId]["MultiDay_Begin_TimeRelevant"] = {}
            
        events[corpusFile][eId]["MultiDay_Begin_TimeRelevant"][tId] = None
        
    if correctLabel != 'n/a':
        numLabels += 1
        numCorrectLabels += 1 if correctLabel == predLabel else 0
        
        if predLabel not in freqDist:
            freqDist[predLabel] = {}
        if correctLabel not in freqDist[predLabel]:
            freqDist[predLabel][correctLabel] = 0
            
        freqDist[predLabel][correctLabel] += 1

print freqDist
print "Accuracy for "+eventTypePath+": ", float(numCorrectLabels) / numLabels

        
# :: Read MultiDay Begin TimexRelations ::  
numLabels = 0
numCorrectLabels = 0      
freqDist = {}


eventTypePath = 'output/3_MultiDay/3_Begin_TimexRelations/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    tId = splits[2]
    predLabel = splits[predLabelPosition].lower()
    correctLabel = splits[-2].lower()
    
    if "MultiDay_Begin_TimeRelevant" in events[corpusFile][eId] and tId in events[corpusFile][eId]["MultiDay_Begin_TimeRelevant"]:        
        events[corpusFile][eId]["MultiDay_Begin_TimeRelevant"][tId] = predLabel
  
    if correctLabel != 'n/a':
        numLabels += 1
        numCorrectLabels += 1 if correctLabel == predLabel else 0
        
        if predLabel not in freqDist:
            freqDist[predLabel] = {}
        if correctLabel not in freqDist[predLabel]:
            freqDist[predLabel][correctLabel] = 0
            
        freqDist[predLabel][correctLabel] += 1

print freqDist
print numLabels
print "Accuracy for "+eventTypePath+": ", float(numCorrectLabels) / numLabels



# :: Read MultiDay Begin Relevant TimeValues ::
eventTypePath = 'output/3_MultiDay/4_End_TimeIsRelevant/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    tId = splits[2]
    predLabel = splits[predLabelPosition].lower()
    
    if predLabel == 'relevant':    
        if 'MultiDay_End_TimeRelevant' not in events[corpusFile][eId]:
            events[corpusFile][eId]["MultiDay_End_TimeRelevant"] = {}
            
        events[corpusFile][eId]["MultiDay_End_TimeRelevant"][tId] = None
        
# :: Read MultiDay Begin TimexRelations ::        
eventTypePath = 'output/3_MultiDay/5_End_TimexRelations/output.txt'
for line in open(eventTypePath):
    splits = line.strip().split('\t')
    
    corpusFile = splits[0]
    eId = splits[1]
    tId = splits[2]
    predLabel = splits[predLabelPosition].lower()
    
    
    if "MultiDay_End_TimeRelevant" in events[corpusFile][eId] and tId in events[corpusFile][eId]["MultiDay_End_TimeRelevant"]:        
        events[corpusFile][eId]["MultiDay_End_TimeRelevant"][tId] = predLabel
    
    
    
##############################################################
##############################################################
##############################################################
    





# Generate TSV format
eventTypeDistribution = {'singleday': {'singleday': 0, 'multiday': 0}, 'multiday': {'singleday': 0, 'multiday': 0}}

numCorrectSingleLabels = 0
numMutualExclusiveSingleLabels = 0

numCorrectMultiLabels = 0
numCorrectMultiLabelsRelaxedMetric = 0

numCorrectBeginPoints = 0
numCorrectBeginPointsRelaxedMetric = 0

numCorrectEndPoints = 0
numCorrectEndPointsRelaxedMetric = 0

numCorrectLabels = 0
numCorrectLabelsRelaxedMetric = 0

fOut = open('output/output.tsv', 'w')
fOut.write("\t".join(['corpusFile', 'eId', 'eventType', 'SingleDay_DCTRelations', 'correctLabel', 'predLabel', 'equal']))
fOut.write("\n")

for corpusFile in events:
    for eId in events[corpusFile]:
        event = events[corpusFile][eId]
        eventType = event['EventType']
        dctValue = dctValues[corpusFile]
        SingleDay_DCTRelations = event['SingleDay_DCTRelations']
        MultiDay_DCTRelations = event['MultiDay_DCTRelations']
        correctLabel = eventTimes[corpusFile][eId]
        
        correctEventType = 'multiday' if 'begin' in correctLabel or 'end' in correctLabel else 'singleday'

        predLabel = ""
        
        if eventType not in eventTypeDistribution:
            eventTypeDistribution[eventType] = {}
            
        if correctEventType not in eventTypeDistribution[eventType]:
            eventTypeDistribution[eventType][correctEventType] = 0
            
        eventTypeDistribution[eventType][correctEventType] += 1
        
        
        if eventType == 'singleday':  
            
            if SingleDay_DCTRelations == 'dct':
                predLabel = dctValues[corpusFile]
                
            # :: Check, is some time relation simultanous? ::
            else:
                if "SingleDay_TimeRelevant" in event:
                    for tId in event['SingleDay_TimeRelevant']:
                        if event['SingleDay_TimeRelevant'][tId] == 's':
                            timexValue = timexValues[corpusFile][tId]
                            
                            dates = getDates(timexValue)
                            
                            if 'sim' in dates:
                                predLabel = dates['sim']
                            else:
                                predLabel = "after"+dates['after']+"before"+dates['before']
                            break
                        
                if predLabel == "": #Narrow Down to find best match
                    candidatesBefore = []
                    candidatesAfter = []
                    
                    if SingleDay_DCTRelations == 'beforedct':
                        candidatesBefore.append(dctValues[corpusFile])
                    else:
                        candidatesAfter.append(dctValues[corpusFile])
                        
                    
                    
                    predLabel = narrowDown(event, 'SingleDay', candidatesBefore, candidatesAfter)
                    
                
                
                
            # Acc calculation
            if predLabel == correctLabel:
                numCorrectSingleLabels += 1
            elif correctEventType == 'singleday':               
                if mutual_exclusive_score(predLabel, correctLabel):
                    numMutualExclusiveSingleLabels += 1
            
                
        else: #Multi-Day Event
            if corpusFile == 'APW19980227.0494' and eId == 'e46':
                pass


            predBegin = ""
            predEnd = ""
            
            if "MultiDay_Begin_TimeRelevant" in event:
                for tId in event['MultiDay_Begin_TimeRelevant']:
                    if event['MultiDay_Begin_TimeRelevant'][tId] == 's':
                        if len(timexValues[corpusFile][tId]) == 10:
                            predBegin = timexValues[corpusFile][tId]
                            break
                        elif len(timexValues[corpusFile][tId]) == 4:
                            year = timexValues[corpusFile][tId]
                            predBegin = "after"+year+"-01-01before"+year+"-12-31"
                            break
                        
            if "MultiDay_End_TimeRelevant" in event:
                for tId in event['MultiDay_End_TimeRelevant']:
                    if event['MultiDay_End_TimeRelevant'][tId] == 's':
                        #print "End simult: ", timexValues[corpusFile][tId]
                        
                        if len(timexValues[corpusFile][tId]) == 10:
                            predEnd = timexValues[corpusFile][tId]
                            break
                        elif len(timexValues[corpusFile][tId]) == 4:
                            year = timexValues[corpusFile][tId]
                            predEnd = "after"+year+"-01-01before"+year+"-12-31"
                            break
                        
                        
            if predBegin == "": #Narrow Down to find best match
                candidatesBefore = []
                candidatesAfter = []
                
                #if MultiDay_DCTRelations == 'afterdct':
                #    candidatesAfter.append(dctValues[corpusFile])
                #else:
                #    candidatesBefore.append(dctValues[corpusFile])                        
                    
                predBegin = narrowDown(event, 'MultiDay_Begin', candidatesBefore, candidatesAfter)
                
                if len(predBegin) == 0:
                    if MultiDay_DCTRelations == 'afterdct':
                        predBegin = "after"+dctValue 
                    else:
                        predBegin = "before"+dctValue  
                
            
            if predEnd == "": #Narrow Down to find best match
                candidatesBefore = []
                candidatesAfter = []
                
                """
                if MultiDay_DCTRelations == 'beforedct':
                    #candidatesBefore.append(dctValues[corpusFile])    
                    
                    if 'MultiDay_End_TimeRelevant' in event:
                        for tId, tlink in event['MultiDay_End_TimeRelevant'].iteritems():
                            if tlink == 'a':
                                timexValue = timexValues[corpusFile][tId]
                                
                                if timexValue >= dctValues[corpusFile]:
                                    event['MultiDay_End_TimeRelevant'][tId] += 'c'    
                                    pass                                
                else:
                    #candidatesAfter.append(dctValues[corpusFile])
                    pass
                """
                    
                predEnd = narrowDown(event, 'MultiDay_End', candidatesBefore, candidatesAfter)
                        
                if len(predEnd) == 0:
                    if MultiDay_DCTRelations == 'beforedct':
                        predEnd = "before"+dctValue 
                    else:
                        predEnd = "after"+dctValue  
                    
                    
                        
            predLabel = "beginPoint="+predBegin+"endPoint="+predEnd
            
            
            #hard cases:
            # 'APW19980308.0201' -> "in recent years'
            # 'PRI19980306.2000.1675' -> 'it is the second day'
            # 'APW19980227.0489' -> 'where they lived for 2 1/2 years'
            # 
            corpusFilesBlacklist = ['APW19980308.0201', 'PRI19980306.2000.1675', 'APW19980227.0489']
            if correctEventType == eventType:
                if predLabel != correctLabel and False:
                    if corpusFile not in corpusFilesBlacklist: #Skip hard cases / errors in annotations
                        print "\n\n\n"
                        print "pred: ", predLabel, " vs. correct:", correctLabel
                        
                        if 'MultiDay_Begin_TimeRelevant' in event:
                            print corpusFile
                            print "Begin: ",event['MultiDay_Begin_TimeRelevant']
                            for tId, tlink in event['MultiDay_Begin_TimeRelevant'].iteritems():    
                                if tlink != None:                        
                                    print tlink+" "+timexValues[corpusFile][tId]+" ; ",
                                else:
                                    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NONE at "+corpusFile
                            
                            print ""
                                
                        if 'MultiDay_End_TimeRelevant' in event:
                            print "End: ",event['MultiDay_End_TimeRelevant']
                            for tId, tlink in event['MultiDay_End_TimeRelevant'].iteritems():
                                print tlink+" "+timexValues[corpusFile][tId]+" ; ",
                            
                            print ""
                        
                        print "DCT:", dctValues[corpusFile]
                        print "DCT Rel: ", MultiDay_DCTRelations
                        print "File: ", corpusFile
                        print "eId: ", eId                        
                        pass
                
            if predLabel == correctLabel:
                numCorrectMultiLabels += 1
                
            if correctEventType == eventType:
                correctBeginPoint = correctLabel[11:correctLabel.index('endPoint')]
                correctEndPoint = correctLabel[correctLabel.index('endPoint')+9:]
                
                if predBegin == correctBeginPoint:
                    numCorrectBeginPoints += 1                
                
                if predEnd == correctEndPoint:
                    numCorrectEndPoints += 1
                    
                if not mutual_exclusive_score(predBegin, correctBeginPoint):
                    numCorrectBeginPointsRelaxedMetric += 1     
                    
                if not mutual_exclusive_score(predEnd, correctEndPoint):
                    numCorrectEndPointsRelaxedMetric += 1                    
                    
                if not mutual_exclusive_score(predLabel, correctLabel):
                    numCorrectMultiLabelsRelaxedMetric += 1
                            
                            
              
        
        data = [corpusFile, eId, eventType, SingleDay_DCTRelations, correctLabel, predLabel, str(correctLabel == predLabel)]
 
        
        
        fOut.write("\t".join(data))
        fOut.write("\n")
        
        if predLabel == correctLabel:
            numCorrectLabels += 1

        try:
            if not mutual_exclusive_score(predLabel, correctLabel):
                numCorrectLabelsRelaxedMetric += 1
        except:
            pass
        
        
        
        


print eventTypeDistribution
numTotalLabels = eventTypeDistribution['singleday']['singleday']+eventTypeDistribution['singleday']['multiday']+eventTypeDistribution['multiday']['singleday']+eventTypeDistribution['multiday']['multiday']

print "------ Event Type -----------"
print "Event Type Acc", float(eventTypeDistribution['singleday']['singleday']+eventTypeDistribution['multiday']['multiday'])/(numTotalLabels)
print ""

print "------ Single Day Labels -----------"
print "Correct Single Day Labels: ", numCorrectSingleLabels
print "Acc for correct Single Day Labels: %.1f%%" % ((float(numCorrectSingleLabels)/eventTypeDistribution['singleday']['singleday'])*100)
print "Mutual Exclusive Single Day Labels: ", numMutualExclusiveSingleLabels
print "Acc (relaxed metric): %.1f%%" %  ((1-float(numMutualExclusiveSingleLabels)/eventTypeDistribution['singleday']['singleday'])*100)

numLessPrecise = eventTypeDistribution['singleday']['singleday'] - numMutualExclusiveSingleLabels - numCorrectSingleLabels

print "Pecentage less precise: %.1f%%" % (( numLessPrecise/float(eventTypeDistribution['singleday']['singleday']))*100)
print "Percentage of wrong labels: %.1f%%" %  ((float(numMutualExclusiveSingleLabels)/eventTypeDistribution['singleday']['singleday'])*100)
print ""

print "------ Multi Day Labels -----------"
print "Correct Multi Day Labels: ", numCorrectMultiLabels
print "Acc for correct Multi Day Labels: %.1f%%" % (float(numCorrectMultiLabels)/eventTypeDistribution['multiday']['multiday']*100)
print "Acc for correct Multi Day Labels (Relaxed): %.1f%%" % (float(numCorrectMultiLabelsRelaxedMetric)/eventTypeDistribution['multiday']['multiday']*100)
print ""

print "Correct Begin Points: ", numCorrectBeginPoints
print "Acc for correct Begin Points: %.1f%%" % (float(numCorrectBeginPoints)/eventTypeDistribution['multiday']['multiday']*100)
print "Acc for correct Begin Points (Relaxed): %.1f%%" % (float(numCorrectBeginPointsRelaxedMetric)/eventTypeDistribution['multiday']['multiday']*100)

print "Correct End Points: ", numCorrectEndPoints
print "Acc for correct End Points: %.1f%%" % (float(numCorrectEndPoints)/eventTypeDistribution['multiday']['multiday']*100)
print "Acc for correct End Points (Relaxed): %.1f%%" % (float(numCorrectEndPointsRelaxedMetric)/eventTypeDistribution['multiday']['multiday']*100)
print ""

print "------ Overall Performance -----------"

print "Overall Performance: %.1f%%" % (float(numCorrectLabels)/numTotalLabels*100)
print "Overall Performance (Relaxed): %.1f%%" % (float(numCorrectLabelsRelaxedMetric)/numTotalLabels*100)
print ""

print "DONE"

