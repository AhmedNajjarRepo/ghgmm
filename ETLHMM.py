#coding=utf-8

'''Nom ......... : ETMLHMM.py
Role ........ : Module for Extract, transforming and reading data.
Auteur ...... : Ahmed Najjar
Version ..... : V1.0 du 22/12/2014
Licence ..... : GPL
Paper  .....  : @inproceedings{najjar2015two,
  title={Two-Step Heterogeneous Finite Mixture Model Clustering for Mining Healthcare Databases},
  author={Najjar, Ahmed and Gagn{\'e}, Christian and Reinharz, Daniel},
  booktitle={Data Mining (ICDM), 2015 IEEE International Conference on},
  pages={931--936},
  year={2015},
  organization={IEEE}
}
'''
import pdb
import numpy as np
import random as rd



# Function Name: loaddata
# Role: Read of data file
# Parameter are:
#       - Fichier: The name of file containing data raw,
#       - indQual: List of indices of columns for qualitative variables to read.
#       - indQuan: List of indices of columns for quantitative variables to read.
#       - indMul: List of indices of columns for multivalued variables to read
#       - option: Choice of structuration of the data (if 0 : format var1,var2,...,varn if 1:[[var1],[var2],...,[varn])
# Output:
#       - fndata: Final list of data to be clustered containing numerical, categorical and multivalued variables
#       - nominaldata: List of qualitative data for all objects
#       - numericdata: List of quantiative data for all objetcts
#       - multivaluedata: List of multivalued data for all objects

def loaddata(Fichier,indQual,indQuan,indMul,option):
    f=open(Fichier,'r')
    srcdata=[]
    for line in f:
        line.rstrip('\n')
        srcdata.append(line.split(";"))
    f.close()
    i=0
    n = len(srcdata)
    while (i<n):
        for j in range(len(srcdata[i])):
            if(',' in str(srcdata[i][j])):
                long = len(str(srcdata[i][j]))
                srcdata[i][j] = str(srcdata[i][j])[0:long].split(",")
            else :
                srcdata[i][j] = str(srcdata[i][j])
        i = i+1
    fndata = []
    numericdata = []
    nominaldata = []
    multivaluedata = []
    nlg = len(srcdata)
    i=0
    #pdb.set_trace()
    while(i< nlg):
        Var = list([])
        templist = list([])
        for q in indQual:
            if option ==0:
                if type(srcdata[i][q]) == str:
                    Var.append(srcdata[i][q])
                    templist.append(srcdata[i][q])
                else:
                    Var.append(srcdata[i][q][0])
                    templist.append(srcdata[i][q][0])
            else:
                if type(srcdata[i][q]) == str:
                    Var.append([srcdata[i][q]])
                    templist.append([srcdata[i][q]])
                else:
                    Var.append([srcdata[i][q][0]])
                    templist.append([srcdata[i][q][0]])

        nominaldata.append(templist)
        templist = list([])
        for v in indQuan:
            if type(srcdata[i][v]) == str:
                if option ==0:
                    Var.append(float(srcdata[i][v]))
                    templist.append(float(srcdata[i][v]))
                else:
                    Var.append([float(srcdata[i][v])])
                    templist.append([float(srcdata[i][v])])
            else:
                #ns =0
##                for e in range(len(srcdata[i][v])):
##                    ns = ns+ float(srcdata[i][v][e])
                ns = float(srcdata[i][v][0])
                if option ==0:
                    Var.append(float(ns))
                    templist.append(float(ns))
                else:
                    Var.append([float(ns)])
                    templist.append([float(ns)])
        numericdata.append(templist)
        templist = list([])
        
        for t in indMul:
            if type(srcdata[i][t]) is list:
                Var.append(srcdata[i][t])
                templist.append(srcdata[i][t])
            else:
                Var.append([srcdata[i][t]])
                templist.append([srcdata[i][t]])
        multivaluedata.append(templist)
        fndata.append(Var)
        i = i+1
    del(srcdata)
    return fndata,nominaldata,numericdata,multivaluedata


# Function Name: rlabels
# Role: Read of label file
# Parameter are:
#       - Fichier: File containing object labels,
#       - cd: option of read: 's': hospital stay, 'c': Consultation or Visit, other for pathways
# Output:
#       - diff: object labels list.

def rlabels(Fichier,cd):
    f=open(Fichier,'r')
    for line in f:
        srcdata = line.split(",")
    f.close()
    n= len(srcdata)
    srcdata[0] = srcdata[0].strip('[')
    srcdata[n-1]= srcdata[n-1].strip(']\r\n')
    srcdata = map(int, srcdata)
    if cd == 's' or 'c':
        diff = srcdata
    else:
        sous = [1]*n
        diff =  map(operator.sub, srcdata,sous)
    return diff

# Function Name: getCatgories
# Role: Determine the item dictionary
# Parameter are:
#       - srcdata: List of multivalued variable values,
# Output:
#       - c1: Dictionary of items and their counts

def getCatgories(srcdata):
    c1={}
    for i in srcdata:
        for j in i:
            s=set()
            s.add(j[0:2])
            key=frozenset(s)
            # if the item has appeared before,plus one
            if key in c1:
                c1[key]=c1[key]+1
            else:
                c1[key]=1
    return c1

# Function Name: categtostatevisible
# Role: Determine the 1-item list to use as HMM observed states
# Parameter are:
#       - data: List of multivalued variable values,
# Output:
#       - Fliste: List of items 

def categtostatevisible(data):
    c1 = getCatgories(data)
    lc = c1.keys()
    Fliste =[]
    for i in lc:
        Fliste.append(list(i)[0])
    return Fliste

# Function Name: getCatgories
# Role: For each item matches one integer. 
# Parameter are:
#       - listch: List of items,
# Output:
#       - dicch: Dictionary matches items to integer
def listtodict(listch):
    dicch = {}
    n = len(listch)
    for i in range(n):
        dicch[listch[i]] = i
    return dicch

# Function Name: codageSequence
# Role: Transform list of items sequences to list of integer sequences
# Parameter are:
#       - seqMul: List of sequences of values for multivalued variable
#       - dch: dictionary which transform each item to integer
#       - nterme: Number of characters to consider for items
# Output:
#       - seqcode: Sequences of integers
    
def codageSequence(seqMul,dch,nterme):
    seqcode = []
    n = len(seqMul)
    for i in range(n):
        nm = len(seqMul[i])
        l = []
        for j in range(nm):
            l.append(dch[seqMul[i][j][0:nterme]])
        seqcode.append(l)
    return seqcode

# Function Name: InitParametres
# Role: Initialize the parameters of an HMM
# Parameter are:
#       - numstates: number of hidden states
#       - numobservable: number of observed states
# Output:
#       - InitProbs: initial probability vector
#       - TransitionProbs: Transition probability matrix
#       - EmissionProbs: Emission probability matrix

def InitParametres(numstates,numobservable):
    InitProbs = np.array([0]*numstates, dtype=float)
    TransitionProbs = np.ndarray((numstates,numstates), dtype=float)
    EmissionProbs = np.ndarray((numstates,numobservable), dtype=float)
    rd.seed()       
    for i in range(numstates):
        rd.seed()
        InitProbs[i] = rd.random()
        for j in range(numstates):
            rd.seed()
            TransitionProbs[i,j] = rd.random()
        TransitionProbs[i,:] = TransitionProbs[i,:]/np.sum(TransitionProbs,axis =1)[i]
        for j in range(numobservable):
            rd.seed()
            EmissionProbs[i,j] = rd.random()
        EmissionProbs[i,:] = EmissionProbs[i,:]/np.sum(EmissionProbs,axis =1)[i]
    InitProbs[:] = InitProbs[:]/ sum(InitProbs)
     
    return InitProbs,TransitionProbs,EmissionProbs

    
    
# Function Name: getdatabetweenInd
# Role: Takes a list of values for several variables and makes the list of values for variables between
# two indices
# Parameter are:
#       - srcdata: List of variable values
#       - ind1: First index
#       - ind2: Second index
# Output:
#       - lfinal: List of values for variables between ind1 and ind2 

def getdatabetweenInd(srcdata,ind1,ind2):
    i=0
    n = len(srcdata)
    lfinal =[]
    while (i<n):
        lfinal.append(srcdata[i][ind1:ind2])
        i = i+1
    lfinal = np.array(lfinal,float)
    return lfinal





