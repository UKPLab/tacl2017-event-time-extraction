# -*- coding: utf-8 -*-
import re
import numpy as N
from datetime import timedelta, datetime
from collections import namedtuple
import pylab as pl

def normalizeDate(date, dctStr=None, dctMinus1Str=None, fullNormalization=False):
    # Normalizes different date formats
    if date == "undefined":
        return "n/a"
    
    date = date.replace("beginoint", "beginPoint")
    date = date.replace("startPoint", "beginPoint")
    date = date.replace("begore", "before")
    date = date.replace("endoint", "endPoint")
    date = date.replace("beginPont", "beginPoint")
    date = date.replace('19987-01-31', '1987-01-31')
    date = re.sub(r', +', ',', date)
    date = re.sub('\s+', ' ', date).strip()
    date = date.replace("= ", "=")

    for match in re.finditer('-xx-x[^x]', date):
        date = date.replace(match.group()[:len(match.group())-1], '-xx-xx', 1)

    for match in re.finditer('end[^P]', date):
        date = date.replace(match.group()[:len(match.group())-1], "endPoint", 1)

    for match in re.finditer('1990-08-0[^0-9]', date):
        date = date.replace(match.group()[:len(match.group())-1], "1990-08-0", 1)

    for match in re.finditer('[^1-9]998-01-07', date):
        date = date.replace(match.group()[:len(match.group())-1], "1998-01-07", 1)

    date = date.replace('endPointbefore', 'endPoint=before')
    
    if fullNormalization:
        date = date.replace("assumed ", "") 
        date = date.replace(' ', '') 
    
        #normalize xx short form cases
        for match in re.finditer("[0-9]{4}-[0-9]{2}-xx", date):
            date = date.replace(match.group(), "after" + match.group()[:8] + "01before" + match.group()[:8] + "28", 1)
    
        for match in re.finditer("[0-9]{4}-xx-xx", date):
            date = date.replace(match.group(), "after" + match.group()[:5] + "01-01before" + match.group()[:5] + "12-31", 1)
    
        #normalize season short forms
        #After statements
        for match in re.finditer("after([0-9]{4})-FA", date):
            date = date.replace(match.group(), "after" + match.group(1) + "-09-23", 1)
            
        for match in re.finditer("after([0-9]{4})-SP", date):
            date = date.replace(match.group(), "after" + match.group(1) + "-03-20", 1)
            
        for match in re.finditer("after([0-9]{4})-WI", date):
            date = date.replace(match.group(), "after" + match.group(1) + "-12-22", 1)
            
        for match in re.finditer("after([0-9]{4})-SU", date):
            date = date.replace(match.group(), "after" + match.group(1) + "-06-21", 1)
            
        #Before Statements
        for match in re.finditer("before([0-9]{4})-FA", date):
            date = date.replace(match.group(), "before" + match.group(1) + "-12-21", 1)
            
        for match in re.finditer("before([0-9]{4})-SP", date):
            date = date.replace(match.group(), "before" + match.group(1) + "-06-20", 1)
            
        for match in re.finditer("before([0-9]{4})-WI", date):
            date = date.replace(match.group(), "before" + str(int(match.group(1))+1) + "-03-19", 1)
            
        for match in re.finditer("before([0-9]{4})-SU", date):
            date = date.replace(match.group(), "before" + match.group(1) + "-09-22", 1)
            
            
            
        for match in re.finditer("[0-9]{4}-FA", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "09-23endPoint=" + match.group()[:5] + "12-21", 1)
    
        for match in re.finditer("[0-9]{4}-SP", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "03-20endPoint=" + match.group()[:5] + "06-20", 1)
    
        for match in re.finditer("[0-9]{4}-WI", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "12-22endPoint=" + str(int(match.group()[:4])+1) + "-03-19", 1)
    
        for match in re.finditer("[0-9]{4}-SU", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "06-21endPoint=" + match.group()[:5] + "09-22", 1)
    
        for match in re.finditer("[0-9]{4}-02-31", date):
            date = date.replace(match.group(), match.group()[0:8] + "28", 2)
    
        for match in re.finditer("[0-9]{4}-(04|06|09|11)-31", date):
            date = date.replace(match.group(), match.group()[0:8] + "30", 2)
    
        #normalize quarter short forms
        for match in re.finditer("[0-9]{4}-Q1", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "01-01endPoint=" + match.group()[:5] + "03-31", 1)
    
        for match in re.finditer("[0-9]{4}-Q2", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "04-01endPoint=" + match.group()[:5] + "06-30", 1)
    
        for match in re.finditer("[0-9]{4}-Q3", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "07-01endPoint=" + match.group()[:5] + "09-30", 1)
    
        for match in re.finditer("[0-9]{4}-Q4", date):
            date = date.replace(match.group(), "beginPoint=" + match.group()[:5] + "10-01endPoint=" + match.group()[:5] + "12-31", 1)
    
    
        #Converting of the DCT-1 and DCT date to the according time information
        if dctMinus1Str != None:
            date = date.replace("DCT-1", dctMinus1Str)
            date = date.replace("DCT", dctStr)
              
        


    
    return date.strip()
    
def normalizeRealis(realis):
    """
    Normalizes realis values
    """
    realis = realis.lower()
    
    if realis == "webanno.custom.event_":
        return "actual"
        
    elif realis == "generic" or realis == "general":
        return "generic"
    elif realis == "negative" or realis == "negated":
        return "negative"
    
    return realis.lower()

def normalizeTimeX(timeXValue, dctStr=None): #TODO 1998-Wxx and 199X ?!
    """
    Normalizes timex values
    """
    #normalize season short forms
    for match in re.finditer("[0-9]{4}-FA", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "after" + match.group()[:5] + "09-23before" + match.group()[:5] + "12-21", 1)

    for match in re.finditer("[0-9]{4}-SP", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "after" + match.group()[:5] + "03-20before" + match.group()[:5] + "06-20", 1)

    for match in re.finditer("[0-9]{4}-WI", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "after" + match.group()[:5] + "12-22before" + match.group()[:5] + "03-19", 1)

    for match in re.finditer("[0-9]{4}-SU", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "after" + match.group()[:5] + "06-21before" + match.group()[:5] + "09-22", 1)


    #normalize quarter short forms
    for match in re.finditer("[0-9]{4}-Q1", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "beginPoint=" + match.group()[:5] + "01-01endPoint=" + match.group()[:5] + "03-31", 1)

    for match in re.finditer("[0-9]{4}-Q2", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "beginPoint=" + match.group()[:5] + "04-01endPoint=" + match.group()[:5] + "06-30", 1)

    for match in re.finditer("[0-9]{4}-Q3", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "beginPoint=" + match.group()[:5] + "07-01endPoint=" + match.group()[:5] + "09-30", 1)

    for match in re.finditer("[0-9]{4}-Q4", timeXValue):
        timeXValue = timeXValue.replace(match.group(), "beginPoint=" + match.group()[:5] + "10-01endPoint=" + match.group()[:5] + "12-31", 1)

    #normalize Time format (e.g. '1998-03-06TAF)
    if re.match('[0-9]{4}-[0-9]{2}-[0-9]{2}T', timeXValue):
        timeXValue = timeXValue[:10]

    #normalize week numbers (e.g. 1988-W23)
    if re.match('[0-9]{4}-W[0-9]{2}', timeXValue):
        timeXValueSunday = datetime.strptime(timeXValue + '-0', "%Y-W%W-%w")
        timeXValue = timeXValueSunday.strftime("%Y-%m-%d")

    if timeXValue == 'PRESENT_REF' and dctStr != None:
        timeXValue = dctStr

    return timeXValue.strip()

def getDateType(date):
    
    if 'beginPoint' in date or 'endPoint' in date or re.match( r'[0-9]{4}-Q[1-4]{1}', date): 
        return 'timespan'
    
    if 'before' in date and 'after' in date:
        return 'be+af'
    
    if date.startswith('before'):
        return 'before'
        
    if date.startswith('after'):
        return 'after'
        

    if date == 'undefined' or date == 'n/a':
        return 'n/a'
        
    return 'singleD'
    

def getSuperType(date):
    """Coarse grained distinction between n/a, single date events and timespans"""
    if 'beginPoint' in date or 'endPoint' in date or 'Q' in date:
        return 'timespan'
    
    if date == 'undefined' or date == 'n/a':
        return 'n/a'
    
    return "singleDay"

def getBeginEndPoint(timespan):
    """Splits the annotation and returns beginPoint and endPoint as tuple"""
    timeStart = timespan[11:timespan.find("endPoint")]
    timeEnd = timespan[timespan.find("endPoint")+9:]
    return (timeStart, timeEnd)

def getPastPresentFuture(eventTime, dct):
    """Returns whether the expression is in the past, present (DCT), or future"""
    singleDayType = ''
    
    if 'before' in eventTime and 'after' in eventTime:            
        singleDayType = 'after+before'
    elif 'before' in eventTime:
        singleDayType = 'before'
    elif 'after' in eventTime:
        singleDayType = 'after'
    elif re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}", eventTime):
        singleDayType = 'preciseDay'
    
    
    if singleDayType == 'preciseDay':
        date_object = datetime.strptime(eventTime, '%Y-%m-%d')
       
        if date_object < dct:
            return "past"
        elif date_object > dct:
            return "future"
        else:
            return "present"
            
    if singleDayType == 'before':
        dateExpr = eventTime[6:]
        date_object = datetime.strptime(dateExpr, '%Y-%m-%d')
        
        if date_object <= dct:
            return "past"
        else:
            return "other"

           
    if singleDayType == 'after':
        dateExpr = eventTime[5:]
        date_object = datetime.strptime(dateExpr, '%Y-%m-%d')
        
        if date_object >= dct:
            return "future"
        else:
            return "other"

            
    if singleDayType == 'after+before':        
        if 'afterafter' in eventTime or 'beforeafter' in eventTime or ',' in eventTime:
            return "other"
        afterExpr = eventTime[5:15]
        beforeExpr = eventTime[21:]
     
        
        afterObject = datetime.strptime(afterExpr, '%Y-%m-%d')
        beforeObject = datetime.strptime(beforeExpr, '%Y-%m-%d')
        
        if beforeObject <= afterObject:
            #This should never happen
            return "other"
        if beforeObject <= dct:
            return "past"      
        elif afterObject >= dct:
            return "future"
        else:
            return "other"
        
    return "other"


def getBeforeAfter(singleDay):
    """
    Returns after and before for single day events
    """

    if getDateType(singleDay) == 'be+af':
        afterTime = singleDay[5:singleDay.find('before')]
        beforeTime = singleDay[singleDay.find('before') + 6:]

    elif getDateType(singleDay) == 'before':
        afterTime = '1111-01-01'
        beforeTime = singleDay[6:]

    elif getDateType(singleDay) == 'after':
        afterTime = singleDay[5:]
        beforeTime = '9999-12-31'

    elif getDateType(singleDay) == 'singleD':
        afterTime = singleDay 
        beforeTime = singleDay
    else:
        raise ValueError
      
    
    return (afterTime, beforeTime)


def getBeforeAfterTimespan(timespan):
    """
    Returns the after and before for multi day events. It ensures, that it begins
    before it ends and that it ends after it begins
    """
    beginPoint, endPoint = getBeginEndPoint(timespan)
  
    beginAfter, beginBefore = getBeforeAfter(beginPoint)
    endAfter, endBefore = getBeforeAfter(endPoint)
   
    
    
    beginBefore = min(beginBefore, endAfter)
    endAfter = min(endAfter,beginBefore)
    
    return beginAfter, beginBefore, endAfter, endBefore

def isValidAnnotationFormat(date):

    dates = isMultipleDates(date)
    if dates == False:

        if getSuperType(date) == 'n/a':
            return True

        if 'xxxx-xx-xx' in date:
            return True

        if getSuperType(date) == 'singleDay':
            try:
                beforeAfterStr = getBeforeAfter(date)
                datetime.strptime(beforeAfterStr[0], "%Y-%m-%d")
                datetime.strptime(beforeAfterStr[1], "%Y-%m-%d")
                return True

            except ValueError:
                return False


        if getSuperType(date) == 'timespan':
            try:

                beginPointEndPoint = getBeginEndPoint(date)

                beforeAfterBeginPointStr = getBeforeAfter(beginPointEndPoint[0])
                datetime.strptime(beforeAfterBeginPointStr[0], "%Y-%m-%d")
                datetime.strptime(beforeAfterBeginPointStr[1], "%Y-%m-%d")

                beforeAfterBeginPointStr = getBeforeAfter(beginPointEndPoint[1])
                datetime.strptime(beforeAfterBeginPointStr[0], "%Y-%m-%d")
                datetime.strptime(beforeAfterBeginPointStr[1], "%Y-%m-%d")
                return True

            except ValueError:
                return False

    #annotations with multiple dates
    for d in dates:
        if isValidAnnotationFormat(d) == False:
            return False

    return True

def isMultipleDates(date):
    if ',' in date:
        dates = date.split(',')
        return dates

    return False

def nominal_metric(a,b):
    return a != b

def interval_metric(a,b):
    return (a-b)**2

def ratio_metric(a,b):
    return ((a-b)/(a+b))**2


def relaxed_timespan(a, b):
    if a == b:
        return 0
    elif "beginPoint" in a and "beginPoint" in b:
        points1 = getBeginEndPoint(a)
        points2 = getBeginEndPoint(b)
        
        if points1[0] == points2[0] and points1[1] == points2[1]:
            print "!! This should never happen @relaxed_timespan(a,b): !!"
            return 0
        elif points1[0] == points2[0] or points1[1] == points2[1]:
            return 0.5
    return 1


mutalExclusiveCache = {} #Cache results to improve performance
def mutual_exclusive_based_distance(a, b):
    
    if a+";"+b in mutalExclusiveCache:
        return mutalExclusiveCache[a+";"+b]

    if b+";"+a in mutalExclusiveCache:
        return mutalExclusiveCache[b+";"+a]
    
    mutalExclusiveCache[a+";"+b] = mutual_exclusive_score(a,b)
    return mutalExclusiveCache[a+";"+b] 
    
def mutual_exclusive_score(a, b):
    if a == b:
        return 0    
    
    if getSuperType(a) == 'n/a' and getSuperType(b) == 'n/a':
        return 0    
    elif getSuperType(a) == 'n/a' or  getSuperType(b) == 'n/a':
        return 1
    
    if 'xxxx-xx-xx' in a or 'xxxx-xx-xx' in b:
        return 1
    
    if '*' in a or '*' in b:
        return 1
    
    #Multiple dates case: should at least one pair has overlap
    isMultipleDatesA = isMultipleDates(a)
    isMultipleDatesB = isMultipleDates(b)
    if isMultipleDatesA and isMultipleDatesB:
        for d in isMultipleDatesA:
            for dd in isMultipleDatesB:
                if contradiction_based_distance(d, dd) == 0:
                    return 0
        return 1

    if isMultipleDatesA and not isMultipleDatesB:
        for d in isMultipleDatesA:
            if contradiction_based_distance(d, b) == 0:
                return 0
        return 1

    if isMultipleDatesB and not isMultipleDatesA:
        for d in isMultipleDatesB:
            if contradiction_based_distance(d, a) == 0:
                return 0
        return 1
    
    if getSuperType(a) == getSuperType(b) == 'singleDay':
        if isOverlapping(a, b):
            return 0
        else:
            return 1
    elif getSuperType(a) == getSuperType(b) == 'timespan':
        beginEndPoint_a = getBeginEndPoint(a)
        beginEndPoint_b = getBeginEndPoint(b)

        if isOverlapping(beginEndPoint_a[0], beginEndPoint_b[0]) and isOverlapping(beginEndPoint_a[1], beginEndPoint_b[1]):
            return 0
        return 1
    else:
        if getSuperType(a) == 'timespan': 
            beginPoint, endPoint = getBeginEndPoint(a)
            singleDay = b
        else:
            beginPoint, endPoint = getBeginEndPoint(b)
            singleDay = a
            
        beginAfter = datetime.strptime(getBeforeAfter(beginPoint)[0], "%Y-%m-%d")
        beginBefore = datetime.strptime(getBeforeAfter(beginPoint)[1], "%Y-%m-%d")
        endAfter = datetime.strptime(getBeforeAfter(endPoint)[0], "%Y-%m-%d")
        endBefore = datetime.strptime(getBeforeAfter(endPoint)[1], "%Y-%m-%d")
        
        singleDayAfter = datetime.strptime(getBeforeAfter(singleDay)[0], "%Y-%m-%d")
        singleDayBefore = datetime.strptime(getBeforeAfter(singleDay)[1], "%Y-%m-%d")
        
        if isOverlapping(beginPoint, endPoint): #Begin and EndPoint must be overlapping -> so it could also be a single Day event
            if singleDayBefore < beginAfter or singleDayAfter > endBefore:
                return 1
            else:
                return 0
       
        return 1 

    
    
        
        
    

def contradiction_based_distance(a, b):
    """
    Contradiction based distance by Nazanin
    """
    if a == b:
        return 0

    if getSuperType(a) != getSuperType(b):
        return 1

    if getDateType(a) == getDateType(b) == 'n/a':
        return 0

    if 'xxxx-xx-xx' in a or 'xxxx-xx-xx' in b:
        return 1

    #Multiple dates case: should at least one pair has overlap
    isMultipleDatesA = isMultipleDates(a)
    isMultipleDatesB = isMultipleDates(b)
    if isMultipleDatesA and isMultipleDatesB:
        for d in isMultipleDatesA:
            for dd in isMultipleDatesB:
                if contradiction_based_distance(d, dd) == 0:
                    return 0
        return 1

    if isMultipleDatesA and not isMultipleDatesB:
        for d in isMultipleDatesA:
            if contradiction_based_distance(d, b) == 0:
                return 0
        return 1

    if isMultipleDatesB and not isMultipleDatesA:
        for d in isMultipleDatesB:
            if contradiction_based_distance(d, a) == 0:
                return 0
        return 1


    if getDateType(a) == getDateType(b) == 'timespan':
        beginEndPoint_a = getBeginEndPoint(a)
        beginEndPoint_b = getBeginEndPoint(b)

        if isOverlapping(beginEndPoint_a[0], beginEndPoint_b[0]) and isOverlapping(beginEndPoint_a[1], beginEndPoint_b[1]):
            return 0
        return 1

    #if both are single day
    if isOverlapping(a, b):
        return 0
    return 1


def isOverlapping(singleDate1, singleDate2):
    try:
        a_0 = datetime.strptime(getBeforeAfter(singleDate1)[0], "%Y-%m-%d")
        a_1 = datetime.strptime(getBeforeAfter(singleDate1)[1], "%Y-%m-%d")
        b_0 = datetime.strptime(getBeforeAfter(singleDate2)[0], "%Y-%m-%d")
        b_1 = datetime.strptime(getBeforeAfter(singleDate2)[1], "%Y-%m-%d")
    except:
        return False


    minimum = min(a_1, b_1)
    maximum = max(a_0, b_0)

    if minimum >= maximum:
        return True

    return False

def krippendorff_alpha(data,metric=nominal_metric,force_vecmath=False,convert_items=str,missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: str)
    missing_items: indicator for missing items (default: None)
    Source: http://grrrr.org/data/dev/krippendorff_alpha/
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    maskitems = [missing_items]
    if N is not None:
        maskitems.append(N.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.iteritems()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it,g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict((it,d) for it,d in units.iteritems() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.itervalues())  # number of pairable values
    
    N_metric = (N is not None) and ((metric in (interval_metric,nominal_metric,ratio_metric)) or force_vecmath)
    
    Do = 0.
    for grades in units.itervalues():
        if N_metric:
            gr = N.array(grades)
            Du = sum(N.sum(metric(gr,gri)) for gri in gr)
        else:
            Du = sum(metric(gi,gj) for gi in grades for gj in grades)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    De = 0.
    for g1 in units.itervalues():
        if N_metric:
            d1 = N.array(g1)
            for g2 in units.itervalues():
                De += sum(N.sum(metric(d1,gj)) for gj in g2)
        else:
            for g2 in units.itervalues():
                De += sum(metric(gi,gj) for gi in g1 for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De

def find_nearest(array,value):
    idx = (N.abs(array-value)).argmin()
    return array[idx]

# ---- Read in DCT ----
def getDCT(dctFile):
    dct = {}
    for line in open(dctFile):
        splits = line.strip().split()
        docName = splits[0]
        date = splits[1]

        dct[docName] = date[0:10]

    return dct

#plot bar cahrt
def plotBarChart(x, y):
    fig = pl.figure()
    ax = pl.subplot(111)
    ax.bar(x, y, width=1)
    pl.show()