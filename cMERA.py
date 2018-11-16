#!/usr/bin/env python
import numpy as np
import pickle
import os,sys,select
import datetime
import src.cMERAcmpsfunctions as cmf
import src.cMERAlib as cmeralib
import matplotlib.pyplot as plt
import math
import argparse
import warnings
import re
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))
#plt.ion()


class cMERA(object):
    """
    a class for simulating a cMERA evolution
    """
    def __init__(self,dtype=complex,cutoff=1,delta=1E-3j,fullint=False,inter=0,ddint=False,invrange=1.0,nwarmup=2,Dmax=16):
        """
        initialize a cMERA evolution for a scalar boson
        Parameters:

        dtype:  type float or type complex (complex)
                the data type of the cMERA/cMPS matrices
        cutoff: float (1.0)
                UV cutoff of the cMERA
        delta:  complex (0.001j)
                the step-size of the evolution
        fullint: bool (False)
                 use the fully interacting entangler ("soft-phi-four") 
        ddint:   bool (False)
                 use a density-density interacting entangler
        inter:   float
                 the interaction strength of the entangler
        invrange: float
                  inverse length scale of the interaction
        nwarmup:  int
                  number of warmups steps
        Dmax:     int
                  maximum bond dimension
        """
        self.dtype=dtype
        self.cutoff=cutoff
        self.delta=delta
        self.fullint=fullint
        self.inter=inter
        self.ddint=ddint
        self.invrange=invrange
        self.Dmax=Dmax
        self.scale=0.0
        self.iteration=0
        
        self.Ql=np.ones((1,1)).astype(self.dtype)
        self.Rl=np.zeros((1,1)).astype(self.dtype)
        self.lam=np.array([1.0])
        self.D=len(self.lam)
        self.truncated_weight=0.0
        self.Gamma=cmeralib.freeBosoncMERA_CMPO(self.cutoff,self.delta)
        if (self.fullint==True) and (abs(self.inter)>1E-10):
            self.Gammaint=cmeralib.fullInteractingBosoncMERA_CMPO(0.1,self.cutoff,self.invrange,self.delta,self.inter,dtype=self.dtype)
        elif (self.ddint==True) and (abs(self.inter)>1E-10):
            self.Gammaint=cmeralib.interactingBosoncMERA_CMPO(self.invrange,self.delta,self.inter)
        else:
            self.Gammaint=[[np.asarray([1.0]),np.asarray([1.0])],[np.asarray([1.0]),np.asarray([1.0])]]
        self.Dmpoint=self.Gammaint[0][0].shape[0]
        self.Dmpo=self.Gamma[0][0].shape[0]
        #do a warmup evolution without any truncation
        #problems can arise if truncation threshold of the evolution is so large that it truncates away any new schmidt values; in this case
        #the state will stay a product state for all times. A way to circumvent this problem is to either choose a smaller truncation threshold
        #or do a warmup run without truncation. This way, the evolution can introduce Schmidt-values above the truncation threshold
        for n in range(nwarmup):
            self.scale+=np.abs(self.delta)
            self.iteration+=1
            self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gamma[0][0])+np.kron(self.Ql,np.eye(self.Dmpo))+np.kron(self.Rl,self.Gamma[1][0])
            self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gamma[0][1])+np.kron(self.Rl,np.eye(self.Dmpo))
            if np.abs(inter)>1E-8:
                if ddint:
                    self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gammaint[0][0])+np.kron(self.Ql,np.eye(self.Dmpoint))#+np.kron(self.Rl,Gammaint[1][0])
                    self.Rl=np.kron(self.Rl,self.Gammaint[1][1])
                if self.fullint:
                    self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gammaint[0][0])+np.kron(self.Ql,np.eye(self.Dmpoint))#+np.kron(self.Rl,Gammaint[1][0])
                    self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gammaint[0][1])+np.kron(self.Rl,np.eye(self.Dmpoint))
        
            self.Ql=self.Ql*np.exp(-np.imag(self.delta))
            self.Rl=self.Rl*np.exp(-np.imag(self.delta)/2.0)
        

        #self.lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.regauge_with_trunc(self.Ql,[self.Rl],dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=1E-14,\
            #                                                                    ncv=40,numeig=6,pinv=1E-200,thresh=1E-10,trunc=1E-16,Dmax=self.Ql.shape[0],verbosity=0)
        #self.Rl=self.Rl[0]

    def doStep(self,delta=None,pinv=1E-20,tol=1E-10,Dthresh=1E-6,trunc=1E-10,Dinc=1,ncv=30,numeig=6,thresh=1E-8):
        """
        do a single evolution step

        Parameters:

        delta:   float (None)
                 step-size; if None, doStep uses the stepsize given during construction of the simulation instance;
                 otherwise, it constructs propagators using the value of the last call (or the value passed during 
                 construction)
        pinv:    float (1E-20)
                 pseudo-inverse parameter for inversion of the Schmidt-values and reduced density matrices
        tol:     float (1E-10):
                 precision parameter for calculating the reduced density matrices during truncation
        Dthresh: float (1E-6)
                 threshold parameter; if the truncated weight of the last truncation is larger than Dthres,
                 the bond dimension D is increased by Dinc; if D is already at its maximally allowed value, 
                 D is not changed
        trunc:   float (1E-10)
                 truncation threshold during regauging; all Schmidt-values smaller than trunc will be removed, irrespective
                 of the maximally allowed bond-dimension
        Dinc:    int (1) 
                 bond-dimension increment
        ncv:     int (30)nn
                 number of krylov vectors to be used when calculating the transfer-matrix eigenvectors during truncation
        numeig:  int (6)
                 number of eigenvector-eigenvalue pairs of the transfer-matrix to be calculated 
        thresh:  float (1E-10)
                 related to printing some warnings; not relevant
        """
        if (delta!=None) and (delta!=self.delta):
            self.delta=delta
            self.Gamma=cmeralib.freeBosoncMERA_CMPO(self.cutoff,self.delta)
            if (self.fullint==True) and (abs(self.inter)>1E-10):
                self.Gammaint=cmeralib.fullInteractingBosoncMERA_CMPO(0.1,self.cutoff,self.invrange,self.delta,self.inter,dtype=self.dtype)
            elif (self.ddint==True) and (abs(self.inter)>1E-10):
                self.Gammaint=cmeralib.interactingBosoncMERA_CMPO(self.invrange,self.delta,self.inter)
            else:
                self.Gammaint=[[np.asarray([1.0]),np.asarray([1.0])],[np.asarray([1.0]),np.asarray([1.0])]]
            self.Dmpoint=self.Gammaint[0][0].shape[0]
            self.Dmpo=self.Gamma[0][0].shape[0]
            
        if (self.truncated_weight>Dthresh) and (self.D<self.Dmax) and (len(self.lam)==self.D):
            self.D+=Dinc

        self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gamma[0][0])+np.kron(self.Ql,np.eye(self.Dmpo))+np.kron(self.Rl,self.Gamma[1][0])
        self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gamma[0][1])+np.kron(self.Rl,np.eye(self.Dmpo))
        if np.abs(self.inter)>1E-8:
            if self.ddint:
                self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gammaint[0][0])+np.kron(self.Ql,np.eye(self.Dmpoint))
                self.Rl=np.kron(self.Rl,self.Gammaint[1][1])
            if self.fullint:            
                self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gammaint[0][0])+np.kron(self.Ql,np.eye(self.Dmpoint))
                self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gammaint[0][1])+np.kron(self.Rl,np.eye(self.Dmpoint))
            #try:
            #    self.lam,self.Ql,self.Rl,Qrtens,Rrtens,rest2=cmf.regauge_with_trunc(self.Ql,[self.Rl],dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=tol,\
            #                                                                        ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh,trunc=trunc,Dmax=self.D,verbosity=0)
            #    self.truncated_weight+=np.sum(rest2)            
            #    self.Rl=self.Rl[0]
            #except TypeError:
            #    pass
            
        self.lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.canonize(self.Ql,[self.Rl],linit=None,rinit=None,maxiter=100000,tol=tol,\
                                                                  ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh,trunc=trunc,Dmax=self.D,verbosity=0)
        self.truncated_weight=np.sum(rest1)
        self.Rl=self.Rl[0]
            
        self.Ql*=np.exp(-np.imag(self.delta))
        self.Rl*=np.exp(-np.imag(self.delta)/2.0)
        self.scale+=np.abs(self.delta)
        self.iteration+=1        
        
    def save(self,filename):
        """
        dump the cMERA instance into a pickle file "filename".pickle
        Parameters:
        ------------------
        filename: str()
                  the filename (without .pickle ending)
        """
        with open(filename+'.pickle','wb') as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls,filename):
        """
        load a simulation from a pickle file
        Parameters:
        ---------------
        filename: str
                  the pickle file

        Returns:
        ---------------
        a cMERA instance holding the loaded simulation
        
        """
        with open(filename,'rb') as f:
            cls=pickle.load(f)
        return cls
    
    def addMonitoringVariables(self,data_accumulator):
        """
        adds some variables to data_accumulator. 
        data_accumulator: dict()
                          used to store variables of the cMERA simulation
                          the function maps the following keys to data and stores it in data_accumulator
                          'lams': the Schmidt-spectrum at the current step
                          'tw':   the last truncated weight
                          'D':    the current bond dimension
        """
        
        if 'lams' not in data_accumulator:
            data_accumulator['lams']=[self.lam]
        else:
            data_accumulator['lams'].append(self.lam)
        if 'tw' not in data_accumulator:
            data_accumulator['tw']=[self.truncated_weight]
        else:
            data_accumulator['tw'].append(self.truncated_weight)
        if 'D' not in data_accumulator:
            data_accumulator['D']=[len(self.lam)]
        else:
            data_accumulator['D'].append(len(self.lam))
        return data_accumulator

    
def calculateExactCorrelators(data_accumulator,scale,cutoff):
    """
    calculates the exact pi-pi correlation function at scale "scale", starting from an initial
    product state with UV-cutoff given by cutoff

    data_accumulator: dict():
                      the pi-pi correlation function is stored at key 'pipi_exact' as an np.ndarray
                      data_accumulator['pipi_exact']=correlation_function; 
    scale:            float
                      the scale at which to calculate the corelator
    cutoff:           float
                      UV-cutoff
    """
    xexact,pipiexact=cmeralib.FreeBosonPiPiCorrelatorExact(scale)
    if 'pipi_exact' not in data_accumulator:
        data_accumulator['pipi_exact']=[xexact/cutoff,pipiexact*cutoff**2]
    else:
        data_accumulator['pipi_exact'].append(pipiexact*cutoff**2)
        
    return data_accumulator


def calculatePiPiCorrelators(data_accumulator,cmera,N1=10,N2=40000,eps1=1E-4,eps2=4E-2):
    """
    calculates the pi-pi correlation function using the cMPS tensors from cmera
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the pi-pi correlation function is stored at key 'pipi' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    N1,eps1:          int,float
    N2,eps2:          int,float
                      the corralation function is calculated at points np.arange(N1)*eps1 and N1*eps1+np.arange(N2)*eps2
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above
    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam

    x=np.append(np.arange(1,N1+1)*eps1,np.arange(2,N2+1)*eps2)
    pipi1,vec1=cmf.PiPiCorr(Qltens,Rltens,np.diag(lamtens**2),eps1,N1,cmera.cutoff,initial=None)
    pipi2,vec2=cmf.PiPiCorr(Qltens,Rltens,np.diag(lamtens**2),eps2,N2,cmera.cutoff,initial=vec1)
    pipi=np.append(pipi1,pipi2[1::])    

    if 'pipi' not in data_accumulator:
        data_accumulator['pipi']=[x,(cmera.scale,pipi)]
    else:
        data_accumulator['pipi'].append((cmera.scale,pipi))

    return data_accumulator

def calculatedPhidPhiCorrelators(data_accumulator,cmera,N1=10,N2=40000,eps1=1E-4,eps2=4E-2):
    """
    calculates the partial phi-partial phi correlation function using the cMPS tensors from cmera
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the partial phi-partial phi correlation function is stored at key 'dphidphi' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    N1,eps1:          int,float
    N2,eps2:          int,float
                      the corralation function is calculated at points np.arange(N1)*eps1 and N1*eps1+np.arange(N2)*eps2
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above

    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam

    x=np.append(np.arange(1,N1+1)*eps1,np.arange(2,N2+1)*eps2)
    dxphidxphi1,vec1=cmf.dxPhidxPhiCorr(Qltens,Rltens,np.diag(lamtens**2),eps1,N1,cmera.cutoff,initial=None)
    dxphidxphi2,vec2=cmf.dxPhidxPhiCorr(Qltens,Rltens,np.diag(lamtens**2),eps2,N2,cmera.cutoff,initial=vec1)
    dxphidxphi=np.append(dxphidxphi1,dxphidxphi2[1::])

    if 'dphidphi' not in data_accumulator:
        data_accumulator['dphidphi']=[x,(cmera.scale,dxphidxphi)]
    else:
        data_accumulator['dphidphi'].append((cmera.scale,dxphidxphi)) 
    return data_accumulator


def calculatePsiObservables(data_accumulator,cmera):
    """
    calculates the observable <psi> using the cMPS tensors from cmera 
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the partial phi-partial phi correlation function is stored at key 'psi' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above

    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam
    
    psi=np.trace(Rltens.dot(np.diag(lamtens)).dot(np.diag(lamtens)))
    
    if 'psi' not in data_accumulator:
        data_accumulator['psi']=[psi]
    else:
        data_accumulator['psi'].append(psi)        
    return data_accumulator

def calculateDensityObservables(data_accumulator,cmera):
    """
    calculates the observable <psi* psi> (particle density) using the cMPS tensors from cmera
    and stores it in data_accumulator
    Parameters:
    ----------------------
    data_accumulator: dict():
                      the partial phi-partial phi correlation function is stored at key 'density' as an np.ndarray
                      data_accumulator['pipi']=correlation_function; 
    cmera:            cMERA instance
                      a cMERA simulation
    Returns:
    ----------------------
    data_accumulator: dict()
                      see above

    """
    
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam
    
    dens=np.trace(Rltens.dot(np.diag(lamtens)).dot(np.diag(lamtens)).dot(herm(Rltens)))
    
    if 'density' not in data_accumulator:
        data_accumulator['density']=[dens]
    else:
        data_accumulator['density'].append(dens)        
        data_accumulator['psi'].append(psi)        
    return data_accumulator

def checkWicksTheorem(data_accumulator,cmera,N=20000,eps=0.01):
    Qltens=cmera.Ql
    Rltens=cmera.Rl
    lamtens=cmera.lam
    
    x=np.arange(1,N+1)*eps
    wick=cmeralib.checkWickTheorem(Qltens,Rltens,np.diag(lamtens**2),eps,N)
    if 'wick_theorem' not in data_accumulator:
        data_accumulator['wick_theorem']=[x,wick]
    else:
        data_accumulator['wick_theorem'].append(wick)
    return data_accumulator

def convert(val):

    """
    converts an input str() val into its numeric type
    see the conversion rules below
    
    """
    if val=='True':
        return True
    elif val=='False':
        return False
    elif val=='None':
        return None
    else:
        types=[int,float,str]
    for t in types:
        try:
            return t(val)
        except ValueError:
            pass
        
def read_parameters(filename):
    """
    read parameters from a file "filename"
    the file format is assumed to be 

    parameter1 value1
    parameter2 value2
        .
        .
        .

    or 

    parameter1: value1
    parameter2: value2

    Returns:
    python dict() containing mapping parameter-name to its value
    """
    
    params={}
    with open(filename, 'r') as f:
        for line in f:
            if '[' not in line:
                params[line.replace(':','').split()[0]]=convert(line.replace(':','').split()[1])
            else:
                s=re.sub('[\[\]:,\']','',line).split()
                params[s[0]]=[convert(t) for t in s[1::]]
    return params



def plot(data_accumulator,title='',which=('pipi','dphidphi','lam','density','psi','tw','wick')):
    """
    plots data from data_accumulator
    Parameters:
    ---------------------------------
    data_accumulator:  dict()
                       holds the data to be plotted. The data has to be obtained from one of the data-generating
                       functions above
    title:             str()
                       an optional title for the plots
    which:             tuple() of str()
                       the names of the data to be plotted. Each element in which has to be a key
                       in data_accumulator. 
    """
    plt.ion()
    if ('pipi' in which) and ('exact' not in which):
        try:
        
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)
            plt.loglog(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b')
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['numerical evolution'],loc='best')    
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
            
        except KeyError:
            pass
        
    if ('pipi' not in which) and ('exact' in which):
        try:
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)
            plt.loglog(data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['exact free'],loc='best')    
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)

        except KeyError:
            pass
            
    if ('pipi' in which) and ('exact'in which):
        try:
            plt.figure(1)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)
            plt.loglog(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b',data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pipi'][0],np.abs(data_accumulator['pipi'][-1][1]),'-b',data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['numerical evolution','exact free'],loc='best')    
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)

        except KeyError:
            pass
        
    if 'dphidphi' in which:
        try:
            plt.figure(2)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)    
            plt.loglog(data_accumulator['dphidphi'][0],np.abs(data_accumulator['dphidphi'][-1][1]),'-b')
            plt.ylabel(r'$|\langle\partial\phi(x)\partial\phi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.ylim([1E-12,100])
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['dphidphi'][0],np.abs(data_accumulator['dphidphi'][-1][1]),'-b')    
            plt.ylabel(r'$|\langle\partial\phi(x)\partial\phi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['numerical evolution','exact free'],loc='best')
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
            
        except KeyError:
            pass
        
    if 'lam' in which:
        try:
            plt.figure(3)
            plt.title(title)
            plt.semilogy(data_accumulator['scale'][-1],[data_accumulator['lams'][-1]],'bx')        
            plt.ylabel(r'$\lambda_i$',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass

    if 'density' in which:
        try:
            plt.figure(4)
            plt.clf()
            plt.title(title)
            plt.plot(data_accumulator['scale'],data_accumulator['density'],'bd',markersize=6)
            plt.ylabel(r'$\langle n\rangle$',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
    
    if 'psi' in which:
        try:
            plt.figure(5)
            plt.clf()
            plt.title(title)
            plt.plot(data_accumulator['scale'],data_accumulator['psi'],'bd',markersize=6)            
            plt.ylabel(r'$\langle \psi\rangle$',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
        
    if 'tw' in which:
        try:
            plt.figure(6)
            plt.clf()
            plt.title(title)
            plt.plot(data_accumulator['scale'],data_accumulator['tw'],'bd',markersize=6)            
            plt.ylabel(r'tw',fontsize=20)
            plt.xlabel(r'$scale$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
        
    if 'wick' in which:
        try:
            plt.figure(7)
            plt.clf()
            plt.title(title)
            plt.semilogy(data_accumulator['wick_theorem'][0],np.abs(data_accumulator['wick_theorem'][-1]))
            plt.ylabel(r'violation of wicks th.',fontsize=20)
            plt.xlabel(r'$x$',fontsize=20)
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)
        except KeyError:
            pass
    plt.ioff()
if __name__ == "__main__":

    parser = argparse.ArgumentParser('cMERA.py')
    parser.add_argument('--delta',help='imaginary part of the time step for unitary time evolution (0.001)',type=float,default=0.001)
    parser.add_argument('--Dmax', help='maximal cMPS bond dimension (32)',type=int,default=32)
    parser.add_argument('--nwarmup', help='number of initial warmup-steps without truncation (2); ',type=int,default=2)    
    parser.add_argument('--Dinc', help='bond dimension increment (1)',type=int,default=1)    
    parser.add_argument('--cutoff', help='UV cutoff of the entangler (1.0)',type=float,default=1.0)
    parser.add_argument('--invrange', help='inverse interctionrange (1.0)',type=float,default=1.0)    
    parser.add_argument('--pinv',help='pseudoinver cutoff (1E-20); if chosen too large, severe artifacts will show up',type=float,default=1E-20)
    parser.add_argument('--trunc',help='truncation threshold (1E-10); all schmidt-values below trunc will be discarded, irrespective of Dmax',type=float,default=1E-10)
    parser.add_argument('--Dthresh',help='truncation threshold at which the bond dimension is increased by Dinc (1E-6)',type=float,default=1E-6)    
    parser.add_argument('--thresh',help='threshold for "large-imaginary-eigenvalue" error (1E-10); dont worry about it',type=float,default=1E-10)        
    parser.add_argument('--tol', help='tolerance of eigensolver for finding left and right reduced DM (1E-10)',type=float,default=1E-10)
    parser.add_argument('--imax', help='maximum number of iterations (5000)',type=int,default=5000)
    parser.add_argument('--checkpoint', help='save the simulation every checkpoint iterations for checkpointing (100)',type=int,default=100)
    parser.add_argument('--resume_checkpoint', help='load a checkpointed file and resume simulation',type=str)
    parser.add_argument('--filename', help='filename for output (_interactingBosoncMERA)',type=str,default='_interactingBosoncMERA')
    parser.add_argument('--verbosity', help='output verbosity of the program; choose an integer value {0,1,2,3,4} (1)',type=int,default=1)
    parser.add_argument('--loaddir', help='filename of the simulation to be loaded; the resumed simulation will be stored in filename (see above)',type=str)
    parser.add_argument('--parameterfile', help='read parameters from a given file; each line in the file has to contain the parameter name and its value seperated by a whitespace; values passed by file override values passed by command line',type=str)    
    parser.add_argument('--ending', help='suffix of the file names: Ql+args.ending, Rl+args.ending, lam+args.ending ',type=str)
    parser.add_argument('--numeig', help='number of eigenvector in TMeigs (5)',type=int,default=5)
    parser.add_argument('--ncv', help='number of krylov vectors in TMeigs (40)',type=int,default=40)
    parser.add_argument('--show_plots', nargs='+',help='list of strings from {pipi,exact,dphidphi,density,psi,lams,tw}',type=str,default=[''])
    parser.add_argument('--measurestep', help='calculate observables ever measurestep; if 0, nothing is measured (0)',type=int,default=0)
    parser.add_argument('--measure', nargs='+',help='list of strings from {pipi,exact,dphidphi,density,psi,lams,tw}',type=str,default=[''])    
    parser.add_argument('--inter', help='interaction (0.0)',type=float,default=0.0)
    parser.add_argument('--fullint', help='use fully interacting entangle',action='store_true')    
    parser.add_argument('--ddint', help='use densitt-density interactions in entangler',action='store_true')    
    parser.add_argument('--keepcp', help='keep old checkpoint files of the simulation',action='store_true')
    parser.add_argument('--N1', help='number of points for calculating correlators at distances np.arange(N1**eps1 (10)',type=int,default=10)
    parser.add_argument('--N2', help='number of points for calculating correlators at distances eps1*N1+np.arange(N2)*eps2 (40000)',type=int,default=40000)
    parser.add_argument('--N3', help='number of points for calculating violation of wicks theorem at distances eps3*N3 (20000)',type=int,default=20000)        
    parser.add_argument('--eps1', help='discretization for calculating correlators',type=float,default=1E-4)
    parser.add_argument('--eps2', help='discretization for calculating correlators',type=float,default=1E-2)
    parser.add_argument('--eps3', help='discretization for calculating violation of wicks theorem',type=float,default=1E-2)        

    
    args=parser.parse_args()

    observables=['pipi','dphidphi','lam','density','psi','tw','wick','exact']
    if args.parameterfile!=None:
        parameters=read_parameters(args.parameterfile)
        for k,v in parameters.items():
            if (k!='parameterfile') and (k!='filename'):#read all parameters except filename and parameterfile
                setattr(args,k,v)

    if not all([s in observables for s in args.show_plots]):                
        warnings.warn('cannot plot unknown quantity/quantities {0}'.format(np.asarray(args.show_plots)[[s not in observables for s in args.show_plots]]),stacklevel=2)

    if not all([s in observables for s in args.measure]):                
        warnings.warn('cannot measure unknown quantity/quantities {0}'.format(np.asarray(args.measure)[[s not in observables for s in args.measure]]),stacklevel=2)
        
    if not all([s in args.measure for s in args.show_plots]):                
        warnings.warn('observables {0} are not measured, but are passed to --show_plots; skipping plots'.format(np.asarray(args.show_plots)[[s not in args.measure for s in args.show_plots]]),stacklevel=2)


    if args.show_plots!=[''] and not any([s in args.measure for s in observables]):                
        warnings.warn('some parameters have been passed to --show_plots, but no measurements are specified in --measure; skipping plots ',stacklevel=2)
        
    if (args.measurestep!=0):
        if not any([s in args.measure for s in observables]):
            warnings.warn('a nonzero measurestep --measurestep has been given, but no measurement has been specified in --measure',stacklevel=2)


    date=datetime.datetime.now()
    today=str(date.year)+str(date.month)+str(date.day)
    folder=today+args.filename+'delta_{0}_Dmax{1}_cutoff{2}_inter{3}_invrange{4}'.format(args.delta,args.Dmax,args.cutoff,args.inter,args.invrange)
    filename=args.filename+'delta_{0}_Dmax{1}_cutoff{2}_inter{3}_invrange{4}'.format(args.delta,args.Dmax,args.cutoff,args.inter,args.invrange)
    
    if os.path.exists(folder):
        N=20
        print ('Found folder identical to simulation name {0}. Resuming simulation will overwrite existing data.\nYou have {1} seconds to hit CTRL-C to abort.\n Enter any number to continue now'.format(filename,N))
        for n in range(N,0,-1):
            sys.stdout.write("\rResuming in %i seconds" % n)
            sys.stdout.flush()
            a,b,c=select.select([sys.stdin],[],[],1)
            if sys.stdin in a:
                #catches an enter and resumes immediatly
                input()    
                break
            
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    root=os.getcwd()
    os.chdir(folder)
                
    parameters=vars(args)
    with open('parameters.dat','w') as f:
        for k,v in parameters.items():
            f.write(k + ' {0}\n'.format(v))
    f.close()
        
    #store all initialization parameters in a dict()
    init_params=dict(dtype=complex,cutoff=args.cutoff,delta=args.delta*1.0j,fullint=args.fullint,inter=args.inter,ddint=args.ddint,invrange=args.invrange,nwarmup=args.nwarmup,Dmax=args.Dmax)
    #store all evolution parameters in a dict()
    evolution_params=dict(delta=args.delta*1.0j,pinv=args.pinv,tol=args.tol,Dthresh=args.Dthresh,trunc=args.trunc,Dinc=args.Dinc,ncv=args.ncv,numeig=args.numeig,thresh=args.thresh)
    cmera_sim=cMERA(**init_params)

    data_accumulator=dict(scale=[])
    last_stored=None
    for step in range(args.imax):
        if args.checkpoint!=0 and cmera_sim.iteration>0 and step%args.checkpoint==0:
            if args.keepcp==True:
                last_stored=f'cmera_checkpoint_{cmera_sim.iteration}'
                cmera_sim.save(last_stored)                
            else:
                if (last_stored!=None):# and os.path.exists(last_stored):
                    try:
                        os.remove(last_stored+'.pickle')
                    except OSError:
                        pass
                last_stored=f'cmera_checkpoint_{cmera_sim.iteration}'
                cmera_sim.save(last_stored)                

        cmera_sim.doStep(**evolution_params)
        if (args.measurestep!=0) and (cmera_sim.iteration%args.measurestep==0):
            if 'pipi' in args.measure:
                data_accumulator=calculatePiPiCorrelators(data_accumulator,cmera_sim)
            if 'dphidphi' in args.measure:
                data_accumulator=calculatedPhidPhiCorrelators(data_accumulator,cmera_sim)
            if ('exact' in args.measure):
                data_accumulator=calculateExactCorrelators(data_accumulator,cmera_sim.scale,cmera_sim.cutoff)
            if 'psi' in args.measure:
                data_accumulator=calculatePsi(data_accumulator,cmera_sim)
            if 'density' in args.measure:
                data_accumulator=calculateDensity(data_accumulator,cmera_sim)
            if ('lams' in args.measure) or ('tw' in args.measure):                                
                data_accumulator=cmera_sim.addMonitoringVariables(data_accumulator)
            if 'wick' in args.measure:
                data_accumulator=checkWicksTheorem(data_accumulator,cmera_sim,N=args.N3,eps=args.eps3)
            if any([s in args.measure for s in observables]):                
                data_accumulator['scale'].append(cmera_sim.scale)
                with open('data_accumulator'+'.pickle','wb') as f:
                    pickle.dump(data_accumulator,f)

            if args.show_plots!=['']:
                if any([s in args.measure for s in observables]) and any(s in args.measure for s in args.show_plots):                
                    plot(data_accumulator,title='',which=tuple([s for s in args.show_plots if s in args.measure]))#plot only those that are measured
                else:
                    warnings.warn('nothing to plot; skipping plots',stacklevel=2)
                
        sys.stdout.write('\r iteration %i, D=%i Dmax=%i, tw %.10f' %(cmera_sim.iteration,len(cmera_sim.lam),cmera_sim.Dmax,cmera_sim.truncated_weight))
        sys.stdout.flush()

