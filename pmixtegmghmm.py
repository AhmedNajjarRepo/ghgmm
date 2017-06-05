#coding=utf-8

'''Nom ......... : pmixtegmghmm.py
Role ........ : Module for Gaussian and Multinomial model for data
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
import numpy  as np
import random as rd
import math
import ETLHMM as etl
import multiprocessing as mp
import pdb

# Function Name: disQual
# Role: Compute Hamming distance
# Parameter are:
#       - v1: First value of the qualitative variables
#       - v2: Second value of the qualitative variables
#       - ncl: Number of variables to compare
# Output:
#       - resul: Hamming ditance

def disQual(v1,v2,ncl):
    resul = 0
    for j in range(ncl):
        if(v1[j] != v2[j]):
            resul =resul+1
    return resul

# Function Name: disQual
# Role: Compute Euclidean distance
# Parameter are:
#       - v1: First value of the quantitative variables
#       - v2: Second value of the quantiative variables
# Output:
#       - resul: Euclidean ditance

def disQuan(v1,v2):
    diff = (np.array(v1,float)-np.array(v2,float))
    resul = np.linalg.norm(diff)
    return resul

# Function Name: getMod
# Role: Extract modalities for a given qualitative variable 
# Parameter are:
#       - nominaldata: list of values of the qualitative variables
#       - numvar: Index of the qualitative variable
# Output:
#       - s1: List of modalities for this variables


def getMod(nominaldata,numvar):
    n = len(nominaldata)
    s1 =[]
    for i in range(n):
        if nominaldata[i][numvar] not in s1:
                s1.append(nominaldata[i][numvar])
    return s1

# Function Name: kmixte
# Role: Determines a partition from the mixed data that will be used to initialize the parameters
# of the distributions
# Parameter are:
#       - nominaldata: list of values of the quantitative variables
#       - nomdata: list of values of the qualitative variables
#       - mode: List of modality for qualitative variables
#       - k: number of clusters
# Output:
#       - centre: cluster centers
#       - variance: Variance values 
#       - ncentre: means values
#       - freqinit: Frequencies of modalities
#       - error: clustering error

def kmixte(numericdata,nomdata,mode,k):
    nlg = len(numericdata)
    nc = len(numericdata[0])
    nnc = len(nomdata[0])
    setind = set()
    setvalue = set()
    error = 0
    olderror = 1.797693e308
    niter = 0
    ind = rd.randint(0,nlg-1)
    ind1 = rd.randint(0,nlg-1)
    SommeenCentre = np.zeros((k,nc))
    counts = [0] * k
    labels = [0] * nlg
    nsetind = set()
    nsetvalue = set()
    nerror = 0
    nolderror = 1.797693e308
    ncounts = [0] * k
    nlabels = [0] * nlg 
    ncentre = []
    # choix des individus
    i = 0
    while (i<k):
        #while (ind in setind) or (numericdata[ind][0] in setvalue) or (ind1 in nsetind) or (nomdata[ind][0] in nsetvalue) :
        ind = rd.randint(0,nlg-1)
        ind1 = rd.randint(0,nlg-1)
        setind.add(ind)
        nsetind.add(ind1)
        setvalue.add(numericdata[ind][0])
        nsetvalue.add(nomdata[ind1][0])
        i=i+1
    #pdb.set_trace()
    centre = []
    variance = []
    for j in range(k):
        centre.append(np.array(list(numericdata[setind.pop()])))
        variance.append([0]*nc)
        ncentre.append(list(nomdata[nsetind.pop()]))

    # Pour les données nominales    
    vnomfreq = []
    freqinit = []
    # fréquences pour les variables qualitatives 
    for j in range(nnc):
        vnomfreq.append(np.zeros((k,len(mode[j]))))
        freqinit.append(np.zeros((k,len(mode[j]))))

    #pdb.set_trace()
    while(abs(olderror-error)> 10**-6) and (niter < 20) :
        #pdb.set_trace()
        olderror = error
        error = 0
        labels = [0] * nlg
        counts = [0] * k
        variance = np.zeros((k,nc))
        SommeenCentre = np.zeros((k,nc))
        #variable nominales
        nolderror = nerror
        nerror = 0
        nlabels = [0] * nlg
        ncounts = [0] * k
        for j in range(nnc):
            vnomfreq[j] = np.zeros((k,len(mode[j])))
        for i in range(nlg):
            dis = [0] * k
            ndis = [0] * k
            for l in range(k):
                dis[l] = disQuan(numericdata[i],centre[l])
                ndis[l] = disQual(nomdata[i],ncentre[l],nnc)
            h = dis.index(min(dis))
            nh = ndis.index(min(ndis))
            labels[i] = h
            nlabels[i] = nh
            counts[h] = counts[h]+1
            ncounts[nh] = ncounts[nh]+1
            SommeenCentre[h,:] = SommeenCentre[h,:] + numericdata[i]
            error = error + min(dis)
            ecart = np.array(numericdata[i]) - np.array(centre[h])
            variance[h] = variance[h]+ecart**2
            #variable nominale
            for j in range(nnc):
                vnomfreq[j][nh][mode[j].index(nomdata[i][j])] = vnomfreq[j][nh][mode[j].index(nomdata[i][j])] + 1
            nerror = nerror + min(ndis)
        #pdb.set_trace()   
        for l in range(k):
            if counts[l] !=0:
                centre[l] = SommeenCentre[l,:]/counts[l]
                variance[l] = variance[l]/counts[l]
            else:
                rd.seed()
                ind = rd.randint(0,nlg-1)
                centre[l] = numericdata[ind]
            if ncounts[l] !=0:
                for j in range(nnc):
                    ncentre[l][j] = mode[j][np.ndarray.argmax(vnomfreq[j][l])]
            else:
                rd.seed()
                ind = rd.randint(0,nlg-1)
                ncentre[l] = nomdata[ind]
        #pdb.set_trace()
        #print(error)
        #print(nerror)
        #print(olderror)
        #print(nolderror)
        #print(counts)
        #print(ncounts)
        niter = niter+1

    for j in range(nnc):
        for h in range(k):
            if ncounts[h] != 0:
                freqinit[j][h,:] = vnomfreq[j][h,:]/ncounts[h]
        
    return centre,variance,ncentre,freqinit,error

# Function Name: Calculparametres
# Role: Compute parameters of the distributions given a partition 
# Parameter are:
#       - nominaldata: list of values of the quantitative variables
#       - nomdata: list of values of the qualitative variables
#       - mode: List of modality for qualitative variables
#       - labels: Object labels
#       - k: number of clusters
# Output:
#       - lamdaMulti: Multinomial ditribution parameters values
#       - modeMulti: List of modalities
#       - nmodal: number of modalities
#       - mu: mean
#       - sigma: Standard deviation
#       - minP: Minimum parameter value
#       - counts: Number of objects in clusters

def Calculparametres(numericdata,nomdata,mode,labels,k):
    nlg = len(numericdata)
    nc = len(numericdata[0])
    nnc = len(nomdata[0])
    ncentre = []
    centre = []
    variance = []
    counts = [0]*k
    variance = np.zeros((k,nc))
    centre = np.zeros((k,nc))
    for j in range(k):
        ncentre.append([])

    SommeenCentre = np.zeros((k,nc))
    # Pour les données nominales    
    vnomfreq = []
    freqinit = []
    # fréquences pour les variables qualitatives 
    for j in range(nnc):
        vnomfreq.append(np.zeros((k,len(mode[j]))))
        freqinit.append(np.zeros((k,len(mode[j]))))

    for i in range(nlg):
        SommeenCentre[labels[i],:] = SommeenCentre[labels[i],:] + numericdata[i]
        counts[labels[i]] = counts[labels[i]]+1
        for j in range(nnc):
                vnomfreq[j][labels[i]][mode[j].index(nomdata[i][j])] = vnomfreq[j][labels[i]][mode[j].index(nomdata[i][j])] + 1


    for l in range(k):
        if counts[l] !=0:
            centre[l] = SommeenCentre[l,:]/counts[l]
            for j in range(nnc):
                ncentre[l].append(mode[j][np.ndarray.argmax(vnomfreq[j][l])])
                freqinit[j][l,:] = vnomfreq[j][l,:]/counts[l]
        else:
            rd.seed()
            ind = rd.randint(0,nlg-1)
            centre[l] = numericdata[ind]
            ncentre[l] = nomdata[ind]

    for i in range(nlg):
        ecart = np.array(numericdata[i]) - np.array(centre[labels[i]])
        variance[labels[i]] = variance[labels[i]]+ecart**2

    for l in range(k):
        if counts[l] != 0:
            variance[l] = variance[l]/counts[l]
       
    lamdaMulti = []
    modeMulti = []
    nmodal = []
    minPmul = []
    for l in range(nnc):
        lamdaMulti.append([])
        modeMulti.append([])
        nmodal.append([])
        minPmul.append([])
        modeMulti[l] = mode[l]
        lamdaMulti[l] = freqinit[l]
        nmodal[l] = len(mode[l])
        minPmul[l] = np.ndarray.min(lamdaMulti[l][np.ndarray.nonzero(lamdaMulti[l])])
    minP = min(minPmul)
    #########################Variables numériques########################
    mu = []
    sigma = []
    for c in range(k):
        mu.append([])
        sigma.append([])
        mu[c] =  centre[c]
        sigma[c] = np.matrix(np.diag(variance[c]))
    counts = np.array(counts,float)/sum(counts)
    return lamdaMulti,modeMulti,nmodal,mu,sigma,minP,counts

# Function Name: pnorm
# Role: Compute the multivariate normal distribution
# Parameter are:
#       - x: values vector
#       - m: mean vector
#       - s : sigma (variances/covariances) matrix
# Output:
#       - valeur: probability value 

def pnorm(x, m, s):
        """ 
        Compute the multivariate normal distribution with values vector x,
        mean vector m, sigma (variances/covariances) matrix s
        """
        #pdb.set_trace()
        x = np.array(x)
        xmt = np.matrix(x-m).transpose()
        xm = np.matrix(x-m)
        try:
            sinv = np.linalg.inv(s)
            if np.linalg.det(s) != 0:
                valeur = (2.0*math.pi)**(-len(x)/2.0)*(1.0/math.sqrt(np.linalg.det(s)))*math.exp(-0.5*(xm*sinv*xmt))
            else:
                valeur = (2.0*math.pi)**(-len(x)/2.0)*math.exp(-0.5*(xm*sinv*xmt))
        except:
            #print("je suis dans le cas particulier de la matrice singulière")
            valeur = (2.0*math.pi)**(-len(x)/2.0)*math.exp(-0.5*(xm*xmt)) 
        return valeur

# Function Name: pnominal
# Role: Compute the multinomial distribution
# Parameter are:
#       - x: value
#       - c: variable index
#       - mode : List of modalities
#       - lamda: multinomial parameter
#       - minP: Minimum parameter value 
# Output:
#       - valeur: probability value 

def pnominal(x,c,mode,lamda,minP):
    indice = mode.index(x)
    valeur =  lamda[c][indice]
    if valeur == 0:
        valeur = minP
        #valeur = 1e-307
    return valeur

# Function Name: CalculAppartenance
# Role: Compute the individual membership value for a given cluster
# Parameter are:
#       - x: numerical and categorical individual values 
#       - c1: number of numerical variables
#       - c2: number of categorical variables
#       - c: cluster index
#       - proba: proportion for the clusters
#       - modeMulti : List of modalities
#       - lamdaMulti: multinomial parameter
#       - sigma: sigma (variances/covariances) matrix
#       - mu: mean
#       - minP: Minimum parameter value
#       - lis: List containing the indices of the individuals (used for parallel computing)
#       - R: Queue-communication channel between processes (used for parallel computing)
# Output:
#       - valeur: probability value 

def CalculAppartenance(x,c1,c2,c,proba,modeMulti,lamdaMulti,sigma,mu,minP,lis,R):
    ni = len(lis)
    P = np.ones(ni)
    for i in range(ni):
        for l in range(c1):
            P[i] = P[i]*pnominal(x[lis[i]][l],c,modeMulti[l],lamdaMulti[l],minP)
        P[i] = P[i]*pnorm(x[lis[i]][c1:c1+c2], mu[c], sigma[c])
        P[i] = proba[c]* P[i]
        if P[i] == 0:
            P[i] = minP
    R.put([P,lis])

# Function Name: paramInitiale
# Role: Compute the initial parameters of the distributions
# Parameter are:
#       - x: numerical and categorical individual values
#       - nomdata: list of values of the qualitative variables
#       - numericdata: list of values of the quantitative variables
#       - mode: List of modalities
#       - k: cluster index
#       - c1: number of numerical variables
#       - c2: number of categorical variables
# Output:
#       - lamdaMulti: multinomial parameter
#       - modeMulti : List of modalities
#       - nmodal: number of modalities
#       - mu: mean
#       - sigma: sigma (variances/covariances) matrix
#       - minP: Minimum parameter value

def paramInitiale(x,nomdata,numericdata,mode,k,c1,c2):
    ##############################Variable nominales#######################
    n = len(x)
    lamdaMulti = []
    modeMulti = []
    nmodal = []
    minPmul = []
    error = 0
    minerror = 1.797693e308
    nrep = 0
    maxrep = 8
    while (nrep< maxrep):
        fcentre,fvariance,fmcentre,ffreqinit,error = kmixte(numericdata,nomdata,mode,k)
        #pdb.set_trace()
        if error < minerror:
            centre = fcentre
            variance = fvariance
            mcentre = fmcentre
            freqinit = ffreqinit
            minerror = error
        nrep = nrep+1
    for l in range(c1):
        lamdaMulti.append([])
        modeMulti.append([])
        nmodal.append([])
        minPmul.append([])
        modeMulti[l] = mode[l]
        lamdaMulti[l] = freqinit[l]
        nmodal[l] = len(mode[l])
        minPmul[l] = np.ndarray.min(lamdaMulti[l][np.ndarray.nonzero(lamdaMulti[l])])
    minP = min(minPmul)
    #########################Variables numériques########################
    mu = []
    sigma = []
    for c in range(k):
        mu.append([])
        sigma.append([])
        mu[c] =  centre[c]
        sigma[c] = np.matrix(np.diag(variance[c]))
    ######################################################################
    return lamdaMulti,modeMulti,nmodal,mu,sigma,minP

# Function Name: kMixtureModelGauNom
# Role: Clustering mixed data
# Parameter are:
#       - x: numerical and categorical individual values
#       - k: cluster index
#       - c1: number of numerical variables
#       - c2: number of categorical variables
#       - modeMulti : List of modalities
#       - nmodal: number of modalities
#       - proba: proportion for the clusters
#       - lamdaMulti: multinomial parameter values
#       - mu: mean
#       - sigma: sigma (variances/covariances) matrix
#       - minP: Minimum parameter value

# Output:
#       - Pclust: membership values
#       - Px: probability matrix
#       - labels: Object labels
#       - proba: proportion for the clusters
#       - lamdaMulti: multinomial parameter values
#       - mu: mean
#       - sigma: sigma (variances/covariances) matrix
#       - logvraisemblance: loglikelihood value


def kMixtureModelGauNom(x,k,c1,c2,modeMulti,nmodal,proba,lamdaMulti,mu,sigma,minP):
    n = len(x)
    maxiter = 50
    numjob = 8
    niter = 0
    epsilon = 10**-6
    Px = np.ndarray([n,k], np.float64)  # f(x,alpha)
    Pclust = np.ndarray([n,k], np.float64) # Tik
    oldlogvraisemblance = 0
    logvraisemblance = 1
    Gx = etl.getdatabetweenInd(x,c1,c1+c2) 
    while ((abs(logvraisemblance-oldlogvraisemblance)/abs(logvraisemblance)>=epsilon) and (niter < maxiter)):
        oldlogvraisemblance = logvraisemblance
        Px = np.ones((n,k))
        Pclust= np.ones((n,k))
        # Etape d'espérance
        #print(" je suis à l'étape de l'espérance")
        listi = []
        for g in range(numjob-1):
            listi.append(range(g*(n/numjob),(g+1)*(n/numjob)))
        listi.append(range((numjob-1)*(n/numjob),numjob*(n/numjob)+n%numjob))
        for c in range(k):
            #print("je suis dans la classe %d"%c)
            res   = mp.Queue()
            mint  = [mp.Process(target=CalculAppartenance,args=(x,c1,c2,c,proba,modeMulti,lamdaMulti,sigma,mu,minP,listi[i],res)) for i in range(numjob)]
            for i in mint:
                i.start()
            P,lis = res.get()
            #pdb.set_trace()
            Px[lis,c] = np.array(P[0:len(lis)])
            comp =numjob-1
            while comp !=0:
                P,lis = res.get()
                Px[lis,c] = np.array(P[0:len(lis)])
                comp -=1
            for i in mint:
                i.terminate()
        #print("parallelisation reussite")
        minP = np.ndarray.min(Px)
        proba  = np.array([0]*k)         
        sumPx = np.array([0]*k)
        for i in range(n):
            if sum(Px[i,:]) != 0:
                Pclust[i,:] = Px[i,:]/sum(Px[i,:])
            sumPx = sumPx + Pclust[i,:]
            proba = proba +Pclust[i,:]
        labels = [0]*n
        for i in range(n):
            try:
                labels[i] = list(Pclust[i,:]).index(max(Pclust[i,:]))
            except:
                labels[i] = rd.randint(0,k-1)
                print(i)
                print(Px[i,:])
        #labels = [list(Pclust[i,:]).index(max(Pclust[i,:])) for i in range(n)]
        #print(" je suis à l'étape maximisation")
        for l in range(c1):
            lamdaMulti[l] = np.zeros((k,nmodal[l]))   
        for c in range(k):
            proba[c] = proba[c]/n
        
        for c in range(k):
            mu[c] =  np.array([0]*c2,float)
            sigma[c] = np.matrix(np.diag([0]*c2))
            for l in range(c2):
                if sum(Px[i,:]) != 0:
                    mu[c][l] = sum(Pclust[:,c]*Gx[:,l])/float(sumPx[c])    
        for i in range(n):
            for c in range(k):
                sigma[c] = sigma[c] + Pclust[i,c]*(np.matrix(Gx[i,:]-mu[c]).transpose()*np.matrix(Gx[i,:]-mu[c]))
                for l in range(c1):
                    h = modeMulti[l].index(x[i][l])
                    lamdaMulti[l][c][h] = lamdaMulti[l][c][h] + Pclust[i,c]
                                                   
        for c in range(k):
            if sum(Px[i,:]) != 0:
                sigma[c] = sigma[c]/float(sumPx[c])
                sigma[c] = np.matrix(np.diag(np.diag(sigma[c])))

        for l in range(c1):
            for c in range(k):
                if sum(Px[i,:]) != 0:
                    lamdaMulti[l][c,:] = lamdaMulti[l][c,:]/float(sumPx[c])
                indice  = [o for o,v in enumerate(lamdaMulti[l][c,:]) if lamdaMulti[l][c,:][o] == 0]
                if indice:
                    for b in indice:
                        lamdaMulti[l][c][b] = 1
                        
        logvraisemblance = 0
        for i in range(n):
            for c in range(k):
                if Px[i,c] != 0:
                        logvraisemblance = logvraisemblance+ Pclust[i,c]* np.log(Px[i,c])
        #print(" la logvraisemblance à l'étape %d est %.5f"%(niter,logvraisemblance))
        niter = niter+1
    if (niter == maxiter):
        print("maximum iteration atteint")
        print(niter)
    else:
        print("convergence")
        print(abs(logvraisemblance-oldlogvraisemblance)/abs(logvraisemblance))
        print(logvraisemblance)
        print(oldlogvraisemblance)
        print(niter)
    return Pclust,Px,labels,proba,lamdaMulti,mu,sigma,logvraisemblance

