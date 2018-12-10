#!/usr/bin/env python
import numpy as np
import pickle
import sys
import src.cMERAcmpsfunctions as cmf
import src.cMERAlib as cmeralib
import src.utils as utils
import matplotlib.pyplot as plt
import math
import warnings
import re
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))
class cMERA(object):
    """
    a class for simulating a cMERA evolution
    """
    def __init__(self,cutoff=1.0,alpha=None,inter=0.0,invrange=1.0,operators=['n','n'],delta=1E-3j,nwarmup=2,Dmax=16,dtype=complex):
        """
        initialize a cMERA evolution for a scalar boson
        Parameters:
        --------------------
        cutoff:    float (1.0)
                   UV cutoff of the cMERA
                   cutoff enters in the definition of the free entangler K0 (see below) 
                   as well as in the definition of the interacting entangler via the definition of 
                   psi(x) =\sqrt(cutoff/2)phi(x)+1/sqrt(2*cutoff)pi(x)
        alpha:     float or None
                   prefactor of the free entangler, if None, alpha=cutoff/4 is used
                   the free entangler has the form 
                   K0=alpha\int dx dy exp(-cutoff*abs(x-y)) :pi(x)phi(y):
                     =-1j*alpha/2\int dx dy exp(-cutoff*abs(x-y))*(psi(x)psi(y)-psidag(x)psidag(y))
                   to obtain the correct entangler for the ground-state of free boson with cutoff, use alpha=cutoff/4 (default)
        inter:     float
                   the interaction strength of the entangler
        invrange:  float
                   inverse length scale of the interaction
        operators: list of str
                   operators used to construct entangling propagator
                   case 1: for operators = ['n','n'], the entangler is given by inter * \int dx dy n(x) n(y) exp(-invrange*abs(x-y))
                           if len(operators)==2, operators=['n','n'] is the only allowed choice
                   case 2: for operators = [o1,o2,o3,o4], the entangler is given by inter * \int dw dx dy dx o1(w) o2(x) o3(y) o4(z) exp(-invrange abs(w-z))
                           if len(operators)==4, each element in operators can be one of the following :'phi','pi','psi','psidag'
        delta:     complex 
                   the step-size of the evolution
        nwarmup:   int
                   number of warmup steps; during warmup, no truncation is done
        Dmax:      int
                   maximally allowed bond dimension
        dtype:     type float or type complex (complex)
                   the data type of the cMERA/cMPS matrices

        """
        
        self.dtype=dtype
        self.Dmax=Dmax
        self.scale=0.0
        self.iteration=0
        self.cutoff=cutoff
        self.Ql=np.ones((1,1)).astype(self.dtype)
        self.Rl=np.zeros((1,1)).astype(self.dtype)
        self._lam=np.array([1.0])
        self.D=len(self._lam)
        self.truncated_weight=0.0
        self.Gamma=cmeralib.freeEntanglingPropagator(cutoff=cutoff,delta=delta,alpha=alpha)

        for n in range(nwarmup):
            self.doStep(cutoff=cutoff,alpha=alpha,inter=inter,invrange=invrange,operators=operators,delta=delta,truncAfterFree=False,truncAfterInt=False)
            #do a warmup evolution without any truncation
            #problems can arise if truncation threshold of the evolution is so large that it truncates away any new schmidt values; in this case
            #the state will stay a product state for all times. A way to circumvent this problem is to either choose a smaller truncation threshold
            #or do a warmup run without truncation. This way, the evolution can introduce Schmidt-values above the truncation threshold
        #self.lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.regauge_with_trunc(self.Ql,[self.Rl],dx=0.0,gauge='symmetric',linitial=None,rinitial=None,nmaxit=100000,tol=1E-14,\
            #                                                                    ncv=40,numeig=6,pinv=1E-200,thresh=1E-10,trunc=1E-16,Dmax=self.Ql.shape[0],verbosity=0)
        #self.Rl=self.Rl[0]
        
    def truncate(self,Dmax,trunc=1E-14,tol=1E-12,ncv=30,numeig=6,pinv=1E-200,thresh=1E-8):
        """
        Truncate the cMPS matrices Ql and Rl 
        Parameters:
        -------------------------------------
        Dmax:     int
                  maximum bond dimension
        trunc:    float (1E-14)
                  truncation threshold during regauging; all Schmidt-values smaller than trunc will be removed, irrespective
                  of the maximally allowed bond-dimension
        tol:      float (1E-10):
                  precision parameter for calculating the reduced density matrices during truncation
        ncv:      int (30)nn
                  number of krylov vectors to be used when calculating the transfer-matrix eigenvectors during truncation
        numeig:   int (6)
                  number of eigenvector-eigenvalue pairs of the transfer-matrix to be calculated 
        pinv:     float (1E-20)
                  pseudo-inverse parameter for inversion of the Schmidt-values and reduced density matrices
        thresh:   float (1E-10)
                  related to printing some warnings; not relevant
        """
        
        try:                
            self._lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.canonize(self.Ql,[self.Rl],linit=None,rinit=None,maxiter=100000,tol=tol,\
                                                                       ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh,trunc=trunc,Dmax=Dmax,verbosity=0)
            self.truncated_weight=np.sum(rest1)
            self.Rl=self.Rl[0]
            return rest1            
        except TypeError:
            pass
    
    def doStep(self,cutoff=1.0,alpha=None,inter=0.0,invrange=1.0,operators=['n','n'],delta=1E-3,truncAfterFree=True,truncAfterInt=True,
               pinv=1E-200,tol=1E-12,Dthresh=1E-6,trunc=1E-10,Dinc=1,ncv=30,numeig=6,thresh=1E-8):
        """
        do a single evolution step

        Parameters:
        cutoff:    float (1.0)
                   UV cutoff of the cMERA
                   cutoff enters in the definition of the free entangler K0 (see below) 
                   as well as in the definition of the interacting entangler via the definition of 
                   psi(x) =\sqrt(cutoff/2)phi(x)+1/sqrt(2*cutoff)pi(x)
        alpha:     float or None
                   prefactor of the free entangler, if None, alpha=cutoff/4 is used
                   the free entangler has the form 
                   K0=alpha\int dx dy exp(-cutoff*abs(x-y)) :pi(x)phi(y):
                     =-1j*alpha/2\int dx dy exp(-cutoff*abs(x-y))*(psi(x)psi(y)-psidag(x)psidag(y))
                   to obtain the correct entangler for the ground-state of free boson with cutoff, use alpha=cutoff/4 (default)
        inter:     float
                   the interaction strength of the entangler
        invrange:  float
                   inverse length scale of the interaction
        operators: list of str
                   operators used to construct entangling propagator
                   case 1: for operators = ['n','n'], the entangler is given by inter * \int dx dy n(x) n(y) exp(-invrange*abs(x-y))
                           if len(operators)==2, operators=['n','n'] is the only allowed choice
                   case 2: for operators = [o1,o2,o3,o4], the entangler is given by inter * \int dw dx dy dx o1(w) o2(x) o3(y) o4(z) exp(-invrange abs(w-z))
                           if len(operators)==4, each element in operators can be one of the following :'phi','pi','psi','psidag'
        delta:    float
                  step-size;
                  propagators are constructed using delta
        truncAfterFree:  bool
                         if True, truncate the cMPS after application of the free propagator
                         Application order of propagators: 
                         1. free evolution
                         2. interacting evolution
                         3. rescaling
        truncAfterInt:   bool
                         if True, truncate the cMPS after application of the interacting propagator
                         for operators =['phi','phi','phi','phi'], bond dimension of the propagator 
                         is 4. Together with bond dimension 3 of free propagator, a full application 
                         without intermediate truncation increases D by a factor of 12, which can cause
                         slowness.
                         Application order of propagators: 
                         1. free evolution
                         2. interacting evolution
                         3. rescaling
        pinv:     float (1E-20)
                  pseudo-inverse parameter for inversion of the Schmidt-values and reduced density matrices
        tol:      float (1E-10):
                  precision parameter for calculating the reduced density matrices during truncation
        Dthresh:  float (1E-6)
                  threshold parameter; if the truncated weight of the last truncation is larger than Dthres,
                  the bond dimension D is increased by Dinc; if D is already at its maximally allowed value, 
                  D is not changed
        trunc:    float (1E-10)
                  truncation threshold during regauging; all Schmidt-values smaller than trunc will be removed, irrespective
                  of the maximally allowed bond-dimension
        Dinc:     int (1) 
                  bond-dimension increment
        ncv:      int (30)nn
                  number of krylov vectors to be used when calculating the transfer-matrix eigenvectors during truncation
        numeig:   int (6)
                  number of eigenvector-eigenvalue pairs of the transfer-matrix to be calculated 
        thresh:   float (1E-10)
                  related to printing some warnings; not relevant
        """
        
        self.cutoff=cutoff        
        if len(operators)==2:
            if not all([o=='n' for o in operators]):
                raise ValueError("unknown operators {}. If len(operators)==2, elements in operators can only be 'n'".format(np.array(operators)[[o != 'n' for o in operators]]))
            if abs(inter)>1E-10:            
                self.Gammaint=cmeralib.density_density_interactingEntanglingPropagator(invrange=invrange,delta=delta,inter=inter)
                self.Dmpoint=self.Gammaint[0][0].shape[0]
                interactiontype='nn'
        elif len(operators)==4:
            if not all([o in ('phi','pi','psi','psidag') for o in operators]):
                print()
                raise ValueError("unknown operators {}. If len(operators)==4, each element in operators has to be one of ('phi','pi','psi','psidag')".format(np.array(operators)[[o not in ('phi','pi','psi','psidag') for o in operators]]))
            if abs(inter)>1E-10:
                self.Gammaint=cmeralib.interactingEntanglingPropagator(cutoff=cutoff,invrange=invrange,delta=delta,inter=inter,operators=operators,dtype=self.dtype)
                self.Dmpoint=self.Gammaint[0][0].shape[0]
                interactiontype='oooo'
        else:
            raise ValueError("length of list 'operators' has to be 2 or 4")
        self.Gamma=cmeralib.freeEntanglingPropagator(cutoff=cutoff,delta=delta,alpha=alpha)
        self.Dmpo=self.Gamma[0][0].shape[0]
        
        if (self.truncated_weight>Dthresh) and (self.D<self.Dmax) and (len(self._lam)==self.D):
            self.D+=Dinc
            
        if (alpha==None) or (np.abs(alpha)>1E-10):
            self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gamma[0][0])+np.kron(self.Ql,np.eye(self.Dmpo))+np.kron(self.Rl,self.Gamma[1][0])
            self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gamma[0][1])+np.kron(self.Rl,np.eye(self.Dmpo))
            if truncAfterFree:
                _=self.truncate(Dmax=self.D,trunc=trunc,tol=tol,ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh)
        if np.abs(inter)>1E-10:
            self.Ql=np.kron(np.eye(self.Ql.shape[0]),self.Gammaint[0][0])+np.kron(self.Ql,np.eye(self.Dmpoint))
            if interactiontype=='nn':
                self.Rl=np.kron(self.Rl,self.Gammaint[1][1])
            elif interactiontype=='oooo':                
                self.Rl=np.kron(np.eye(self.Rl.shape[0]),self.Gammaint[0][1])+np.kron(self.Rl,np.eye(self.Dmpoint))
            if truncAfterInt:
                _=self.truncate(Dmax=self.D,trunc=trunc,tol=tol,ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh)                
        self.Ql*=np.exp(-np.imag(delta))
        self.Rl*=np.exp(-np.imag(delta)/2.0)
        self.scale+=np.abs(delta)
        self.iteration+=1        



        
    @property
    def lam(self):
        if self._lam.shape[0]!=self.Ql.shape[0]:
            if len(self._lam)>3:
                self._lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.canonize(self.Ql,[self.Rl],linit=None,rinit=None,maxiter=100000,tol=1E-12,\
                                                                           ncv=min(40,self.Ql.shape[0]),numeig=6,pinv=1E-200,thresh=1E-8,trunc=1E-12,Dmax=self.D,verbosity=0)
                self.Rl=self.Rl[0]
            else:
                pass
        return self._lam
    @lam.setter
    def lam(self,val):
        self._lam=val
            
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
    def read(cls,filename):
        
        """
        read a simulation from a pickle file
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

    
    def load(self,filename):
        
        """
        load a simulation from a pickle file into the current cMERA class
        overwrites current data
        Parameters:
        ---------------
        filename: str
                  the pickle file

        Returns:
        ---------------
        None
        
        """
        with open(filename,'rb') as f:
            cls=pickle.load(f)
        #delete all attribute of self which are not present in cls
        todelete=[attr for attr in vars(self) if not hasattr(cls,attr)]
        for attr in todelete:
            delattr(self,attr)
            
        for attr in cls.__dict__.keys():
            setattr(self,attr,getattr(cls,attr))

    
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
