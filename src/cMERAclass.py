#!/usr/bin/env python
import numpy as np
import pickle
import sys
import copy
import src.cMERAcmpsfunctions as cmf
import src.cMERAlib as cmeralib
import src.utils as utils
import matplotlib.pyplot as plt
import math
import warnings
import re
import random
from pprint import pprint
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
        pinv:     float (1E-200)
                  pseudo-inverse parameter for inversion of the Schmidt-values and reduced density matrices
        thresh:   float (1E-10)
                  related to printing some warnings; not relevant

        Returns:
        ----------------------------------
        np.ndarray:  the truncated Schmidt values

        """
        
        try:                
            self._lam,self.Ql,self.Rl,Qrtens,Rrtens,rest1=cmf.canonize(self.Ql,[self.Rl],linit=None,rinit=None,maxiter=100000,tol=tol,\
                                                                       ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh,trunc=trunc,Dmax=Dmax,verbosity=0)
            self.truncated_weight=np.sum(rest1)
            self.Rl=self.Rl[0]
            return rest1            
        except TypeError:
            return np.ones((1,10))*100000.0
        
    def canonize(self,tol=1E-12,ncv=30,numeig=6,pinv=1E-200,thresh=1E-8):
        """
        canonize the cMPS into left orthogonal form
        Parameters:
        -------------------------------------
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
        _=self.truncate(Dmax=self.Dmax)
        
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
        pinv:     float (1E-200)
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

    def getOptimizedEntangler(self,cost_fun,cost_fun_params,entangler_param_ranges={'cutoffs':np.linspace(0.5,2.5,10)},delta=0.001j,steps=20,pinv=1E-200,
                              tol=1E-12,Dthresh=1E-6,trunc=1E-12,Dinc=1,ncv=30,numeig=6,thresh=1E-8):
        energy_dict={}
    
        ev_params=dict(inter=0,
                   invrange=1.0,
                   operators=['n','n'],
                   delta=delta,
                   truncAfterFree=True,
                   truncAfterInt=True,                          
                   pinv=pinv,
                   tol=tol,
                   Dthresh=Dthresh,
                   trunc=trunc,
                   Dinc=Dinc,
                   ncv=ncv,
                   numeig=numeig,
                   thresh=thresh)
        
        cutoffs=entangler_param_ranges['cutoffs']
        #this is a bit ugly; find a better way to roll out the nested for loops
        if 'cutoffs' not in entangler_param_ranges:
            cutoffs=np.linspace(0.5,2.5,10)
        
        for cutoff in cutoffs:
            if 'alphas' not in entangler_param_ranges:
                alphas=[cutoff/4]
            else:
                alphas=entangler_param_ranges['alphas']
            for alpha in alphas:
                if 'inters' not in entangler_param_ranges:
                    inters=[0.0]
                else:
                    inters=entangler_param_ranges['inters']
                for inter in inters:
                    if 'invranges' not in entangler_param_ranges:
                        invranges=[1.0]
                    else:
                        invranges=entangler_param_ranges['invranges']
                    for invrange in invranges:
                        sim=copy.deepcopy(self)
                        ev_params['cutoff']=cutoff
                        ev_params['alpha']=alpha
                        ev_params['inter']=inter
                        ev_params['invrange']=invrange
                        for s in range(steps):
                            sim.doStep(**ev_params)
                        energy=cost_fun(sim,**cost_fun_params)
                        energy_dict[energy]=dict(cutoff=cutoff,alpha=alpha,inter=inter,invrange=invrange) 
                            
        argmin=energy_dict[np.min(np.array(list(energy_dict.keys())))]
        return argmin,energy_dict



    def _lineSearch(self,cost_fun,cost_fun_params,
                    name='cutoff',test_steps=5,test_delta=0.01j,
                    line_search_params={'start':0.5,'inc':0.02,'maxsteps': 50},
                    other_params={'alpha':None,'inter':0.0,'invrange':1.0},pinv=1E-200,
                    tol=1E-12,Dthresh=1E-6,trunc=1E-12,Dinc=1,ncv=30,numeig=6,thresh=1E-8):
        """
        finds the optimal value of a given parameter of the entangler, as determined by
        cost_fun.

        Parameters:
        -----------------------
        cost_fun:            callable
                             the cost function with respect to which the optimal parameter is found
                             the call signature of cost_fun has to be 
                             val=cost_fun(Ql,Rl,lam,**cost_tun_params)
                             val is of type float
                             Ql,Rl: np.ndarray of shape (D,D)
                             rdens: np.ndarray of shape (D,D)
                             Ql,Rl are left orthogonal cMPS matrices, rdens is the corresponding right reduced density matrix
                             the cMERA class
        cost_fun_params:     dict
                             other parameters of cost_fun
        name:                str
                             parameter name
        test_steps:          int
                             numer of evolution steps carried out before the evaluation of cost_fun
        test_delta:          complex
                             scale increment used for the evolution
        line_search_params:  dict
                             line_search_params['start']=float
                             line_search_params['inc']=int
                             line_search_params['maxsteps']=int
                             initial value, increment and maximum iteration number for the line search
                             parameter_range=np.linspace(param_range)
        pinv:     float (1E-200)
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

        Returns:
        -----------------------------
        (argmin,param_values,energies):
        argmin:       dict
                      maps str to float; keys are str corresponding to the names of the parameters
                      values are the parameter values after optimization
        param_values: dict
                      keys:   str (parameter names)
                      values: np.ndarray: the value of the parameter at each iteration step
        energies:     np.ndarray
                      the energies at each iteration
        """
        if name in other_params.keys():
            raise ValueError('name of optimized parameter also appears on ather_params')
        ev_params=dict(operators=['n','n'],
                       delta=test_delta,
                       truncAfterFree=True,
                       truncAfterInt=True,                          
                       pinv=pinv,
                       tol=tol,
                       Dthresh=Dthresh,
                       trunc=trunc,
                       Dinc=Dinc,
                       ncv=ncv,
                       numeig=numeig,
                       thresh=thresh)
        ev_params.update(other_params)
        found_minimum=False
        energies=[]
        param_values={pname:[] for pname in other_params}
        param_values.update({name:[]})
        step=0
        param=line_search_params['start']
        inc=line_search_params['inc']
        pop_step=0
        while found_minimum==False:
            sim=copy.deepcopy(self)
            ev_params[name]=param
            for s in range(test_steps):
                sim.doStep(**ev_params)
            sim.canonize()
            energies.append(cost_fun(sim.Ql,sim.Rl,np.diag(sim.lam**2),**cost_fun_params))
            for pname in other_params:
                if (pname=='alpha') and (other_params[pname]==None):
                    if name!='cutoff':
                        param_values[pname].append(other_params['cutoff']/4.0)
                    else:
                        param_values[pname].append(param/4.0)                        
                else:
                    param_values[pname].append(other_params[pname])
                    
            param_values[name].append(param)

            if step==0: 
                param=param+inc
                step+=1                
            elif step==1:
                if energies[-2]>energies[-1]:
                    param+=inc
                    step+=1
                if (energies[-2]<energies[-1]) and (pop_step==0):
                    energies.pop(-1)
                    for pname in param_values:
                        param_values[pname].pop(-1)
                    inc=-inc
                    param+=(2*inc)
                    pop_step+=1
                elif (energies[-2]<energies[-1]) and (pop_step==1):
                    for n in param_values:
                        param_values[n].pop(-1)
                    energies.pop(-1)
                    found_minimum=True
                    break

            else:
                if (energies[-1]>energies[-2]) and (energies[-2]<energies[-3]):
                    for n in param_values:
                        param_values[n].pop(-1)
                    energies.pop(-1)
                    found_minimum=True
                    break
                else:
                    param+=inc
                step+=1
            if step>line_search_params['maxsteps']:
                break
            
        minind=np.argmin(energies)
        argmin={p:v[minind] for p,v in param_values.items()}
        return argmin,param_values,energies
    
    def optimizeParameter(self,cost_fun=cmeralib.measure_energy_free_boson_with_cutoff,cost_fun_params={'cutoff':1.0},
                          name='cutoff',
                          delta=0.001j,
                          evo_steps=20,
                          test_delta=0.01j,
                          test_steps=5,
                          line_search_params={'start':0.5,'inc':0.02,'maxsteps':50},
                          precision=0.001,
                          other_parameter_values={'alpha':None,'inter':0.0,'invrange':1.0},
                          maxsteps=4,
                          pinv=1E-200,
                          tol=1E-12,
                          Dthresh=1E-6,
                          trunc=1E-12,
                          Dinc=1,
                          ncv=30,
                          numeig=6,
                          thresh=1E-8,
                          plot=False):


        """
        optimization routine for cMERA class
        finds the optimal value of a given parameter of the entangler, as determined by
        cost_fun, by iteratively evolving the cMERA. The optimization proceeds by 
        first evolving the given cMERA instance over a scale ```S=test_steps*test_delta```, and 
        then evaluating the cost-function. Using a line-search algorithm, the cMERA instance
        is repeatedly evolved up to ```S``` for different values of the parameter ```name```
        (always starting at the same cMERA state), until a minimum of the costfunction ```cost_fun```
        has been found. The resulting minimal value of ```name``` is then used to carry out
        an evolution over a scale ```S_2=delta*evo_steps``` of the initial state of the
        cMERA instance, thereby changing it from its initial value.
        The loop of optimization-evolution is repeated for ```maxsteps``` steps

        Parameters:
        -----------------------
        cost_fun:               callable
                                the cost function with respect to which the optimal parameter is found
                                the call signature of cost_fun has to be 
                                val=cost_fun(Ql,Rl,lam,**cost_fun_params)
                                val is of type float
                                Ql,Rl: np.ndarray of shape (D,D)
                                rdens: np.ndarray of shape (D,D)
                                Ql,Rl are left orthogonal cMPS matrices, rdens is the corresponding right reduced density matrix
        cost_fun_params:        dict
                                other parameters of cost_fun
        name:                   str
                                name of the parameter to be optimized; for gaussian entangler, this can currently be ```name = 'cutoff'``` or ```name = 'alpha'```
        delta:                  complex
                                scale increment used for the evolution carried out after the currently optimal parameter value for parameter ```name``` has been found
        evo_steps:              int
                                numer of evolution steps carried out after the currently optimal parameter value for parameter ```name``` has been found
        test_delta:             complex
                                scale increment used for the test evolution carried out before the evaluation of cost_fun
        test_steps:             int
                                numer of evlution steps carried out before the evaluation of cost_fun
        line_search_params:     dict
                                line_search_params['start']=float
                                line_search_params['inc']=int
                                line_search_params['maxsteps']=int
                                initial value, increment and maximum iteration number for the line search
                                parameter_range=np.linspace(param_range)
        precision:              float
                                optimization stops if parameter is converged within ```precision```
        other_parameter_values: dict
                                dictionary mapping parameter names (str) to values (float or complex);
                                the dictionary contains the values of all non-optimized parameters
        maxsteps:               int
                                number of optimization-evolution steps to be carried out
        pinv:                   float (1E-200)
                                pseudo-inverse parameter for inversion of the Schmidt-values and reduced density matrices
        tol:                    float (1E-10):
                                precision parameter for calculating the reduced density matrices during truncation
        Dthresh:                float (1E-6)
                                threshold parameter; if the truncated weight of the last truncation is larger than Dthres,
                                the bond dimension D is increased by Dinc; if D is already at its maximally allowed value, 
                                D is not changed
        trunc:                  float (1E-10)
                                truncation threshold during regauging; all Schmidt-values smaller than trunc will be removed, irrespective
                                of the maximally allowed bond-dimension
        Dinc:                   int (1) 
                                bond-dimension increment
        ncv:                    int (30)nn
                                number of krylov vectors to be used when calculating the transfer-matrix eigenvectors during truncation
        numeig:                 int (6)
                                number of eigenvector-eigenvalue pairs of the transfer-matrix to be calculated 
        thresh:                 float (1E-10)
                                related to printing some warnings; not relevant

        Returns:
        -----------------------------
        (opt_param_values,accumulated_param_values,energies)

        opt_param_values:         dict
                                  maps str to float; keys are str corresponding to the names of the parameters
                                  values are the parameter values after optimization
        accumulated_param_values: dict
                                  keys:   str (parameter names)
                                  values: np.ndarray: the value of the parameter at each iteration step
        energies:                 np.ndarray
                                  the energies at each iteration
        """



        opt_param_values={name:None}
        opt_param_values.update(other_parameter_values)
        search_params=copy.deepcopy(line_search_params)
        previous_value=1E10
        accumulated_energies=[]
        accumulated_param_values={name:[]}
        accumulated_param_values.update({p:[] for p in other_parameter_values.keys()})
        for step in range(maxsteps):
            argmin,param_values,energies=self._lineSearch(cost_fun=cost_fun,cost_fun_params=cost_fun_params,
                                                          name=name,
                                                          test_steps=test_steps,
                                                          test_delta=test_delta,
                                                          line_search_params=search_params,
                                                          other_params=other_parameter_values,
                                                          pinv=pinv,tol=tol,Dthresh=Dthresh,trunc=trunc,
                                                          Dinc=Dinc,ncv=ncv,numeig=numeig,thresh=thresh)
            for n,v in param_values.items():
                accumulated_param_values[n].extend(v)
            accumulated_energies.extend(energies)
            search_params['start']=argmin[name]
            opt_param_values[name]=argmin[name]
            output={'D':len(self.lam)}
            output.update(opt_param_values)
            if plot:
                if len(accumulated_energies)>1:
                    plt.ion()
                    plt.title(f"optimizing ```{name}```; found mininum at \n {output}")
                    plt.plot(param_values[name],accumulated_energies)
                    plt.legend([name],fontsize=25,loc='best')
                    plt.draw()
                    plt.show()
                    plt.pause(0.01)
                    plt.ioff()
            else:
                print(f'optimized values at step {step}: {output}')
            if np.abs(previous_value-argmin[name])<precision:
                print(f'parameter {name} converged at {argmin[name]} within {precision}')
                break
            else:
                previous_value=argmin[name]
                for s in range(evo_steps):
                    ev_params=dict(operators=['n','n'],
                                   delta=delta,
                                   truncAfterFree=True,
                                   truncAfterInt=True,                          
                                   pinv=pinv,
                                   tol=tol,
                                   Dthresh=Dthresh,
                                   trunc=trunc,
                                   Dinc=Dinc,
                                   ncv=ncv,
                                   numeig=numeig,
                                   thresh=thresh)
                    ev_params.update(opt_param_values)
                    self.doStep(**ev_params)
        return opt_param_values,accumulated_param_values,accumulated_energies


class cMERAoptimizer(object):
    def __init__(self,cmera,name='cMERAoptimizer',cutoff=0.5,alpha=0.7,inter=0.0,invrange=1.0):
        self._parameternames=['cutoff','alpha','inter','invrange']
        self.cmera=copy.deepcopy(cmera)
        self.name=name
        self._it=0
        self.opt_param_values={'cutoff':cutoff,'alpha':alpha,'inter':inter,'invrange':invrange}
        self.accumulated_parameter_values={n:[] for n in self._parameternames}
        self.accumulated_energies=[]
        self._diffs={n:1E10 for n in self._parameternames}
        
    def reset(self,d={'cutoff':random.random(),'alpha':random.random(),'inter':0.0,'invrange':1.0}):
        self._it=0
        self.opt_param_values=copy.deepcopy(d)
        self.accumulated_parameter_values={n:[] for n in self._parameternames}
        self.accumulated_energies=[]
        self._diffs={n:1E10 for n in self._parameternames}

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
        read a optimization from a pickle file
        Parameters:
        ---------------
        filename: str
                  the pickle file
    
        Returns:
        ---------------
        a cMERAoptimizer instance holding the loaded optimization
        
        """
        with open(filename,'rb') as f:
            cls=pickle.load(f)
        return cls

    
    def load(self,filename):
        
        """
        load a simulation from a pickle file into the current cMERAoptimizer class
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


    def optimizeEntangler(self,
                          names=['cutoff','alpha'],
                          maxsteps=20,
                          eps=1E-12,
                          incs=None,
                          maxsteps_linesearch=100,
                          evo_steps0=60,
                          savestep=1,
                          cost_fun=cmeralib.measure_energy_free_boson_with_cutoff,
                          cost_fun_params={'cutoff':1.0},
                          optimizerSteps=4,
                          trunc=1E-10,
                          tol=1E-12,
                          evo_steps=20,
                          test_steps=6,
                          delta=0.001j,
                          test_delta=0.025j,
                          precision=0.001,
                          pinv=1E-200,
                          Dthresh=1E-6,
                          Dinc=1,
                          ncv=30,
                          numeig=6,
                          thresh=1E-8):
        """
        optimize a cMERA entangler

        Parameters:
        ---------------
        names:                  list of str
                                names of the parameters to be optimized
        maxsteps:               int
                                maximum number of outer iteration steps
        eps:                    float
                                desired accuracy of values of the optimized parameters
        incs:                   list or np.ndarray of int, or None
                                len(incs) has to be maxsteps, otherwise an exception is raised
                                for each outermost optimization step ```step```, ```incs[step]```
                                is the fixed increment used in the line search for 
                                finding the minimal value of the currently optimized parameter
        maxsteps_linesearch:    int
                                maximum number of steps after which line search interrupts
        evo_steps0:             int
                                same as evo_steps, but used for the very first outermost iteration 
                                step instead of evo_step
        savestep:               int
                                is ```step%savestep==0```, simulation is checkpointed

        Further parameters which are passed on to cMERA.optimizeParamter:
        ----------------------------------------------------------------------
        cost_fun:               callable
                                the cost function with respect to which the optimal parameter is found
                                the call signature of cost_fun has to be 
                                val=cost_fun(Ql,Rl,lam,**cost_fun_params)
                                val is of type float
                                Ql,Rl: np.ndarray of shape (D,D)
                                rdens: np.ndarray of shape (D,D)
                                Ql,Rl are left orthogonal cMPS matrices, rdens is the corresponding right reduced density matrix
        cost_fun_params:        dict
                                other parameters of cost_fun
        delta:                  complex
                                scale increment used for the evolution carried out after the currently optimal parameter value for parameter ```name``` has been found
        evo_steps:              int
                                numer of evolution steps carried out after the currently optimal parameter value for parameter ```name``` has been found
        test_delta:             complex
                                scale increment used for the test evolution carried out before the evaluation of cost_fun
        test_steps:             int
                                numer of evlution steps carried out before the evaluation of cost_fun
        precision:              float
                                optimization stops if parameter is converged within ```precision```
        optimizerSteps:         int, passed on to maxsteps argument of cMERA.optimizer
                                number of optimization-evolution steps to be carried out
        pinv:                   float (1E-200)
                                pseudo-inverse parameter for inversion of the Schmidt-values and reduced density matrices
        tol:                    float (1E-10):
                                precision parameter for calculating the reduced density matrices during truncation
        Dthresh:                float (1E-6)
                                threshold parameter; if the truncated weight of the last truncation is larger than Dthres,
                                the bond dimension D is increased by Dinc; if D is already at its maximally allowed value, 
                                D is not changed
        trunc:                  float (1E-10)
                                truncation threshold during regauging; all Schmidt-values smaller than trunc will be removed, irrespective
                                of the maximally allowed bond-dimension
        Dinc:                   int (1) 
                                bond-dimension increment
        ncv:                    int (30)nn
                                number of krylov vectors to be used when calculating the transfer-matrix eigenvectors during truncation
        numeig:                 int (6)
                                number of eigenvector-eigenvalue pairs of the transfer-matrix to be calculated 
        thresh:                 float (1E-10)
                                related to printing some warnings; not relevant
                                                        
        """
            
        #save parameter values so we can later retrieve them if neccessary
        self.simulation_params=dict(maxsteps=maxsteps,
                                    optimizerSteps=optimizerSteps,
                                    eps=eps,
                                    trunc=trunc,
                                    tol=tol,
                                    incs=incs,
                                    evo_steps=evo_steps,
                                    test_steps=test_steps,
                                    delta=delta,
                                    test_delta=test_delta,
                                    maxsteps_linesearch=maxsteps_linesearch,
                                    evo_steps0=evo_steps0, 
                                    precision=precision,
                                    pinv=pinv,
                                    Dthresh=Dthresh,
                                    Dinc=Dinc,
                                    ncv=ncv,
                                    numeig=numeig,
                                    thresh=thresh,
                                    savestep=savestep)
        
        with open(self.name+'_parameters.pickle','wb') as f:
            pickle.dump(self.simulation_params,f)
        
        if np.any(incs==None):
            incs=np.ones(maxsteps)*0.000625
            incs[0:min(2,maxsteps)]=0.01
            incs[min(2,maxsteps):min(4,maxsteps)]=0.0075
            incs[min(4,maxsteps):min(6,maxsteps)]=0.005
            incs[min(6,maxsteps):min(8,maxsteps)]=0.0025
            incs[min(8,maxsteps):min(10,maxsteps)]=0.00125
            
        if len(incs)!=maxsteps:
            raise ValueError("length of ```incs``` has to be ```maxsteps```")
        line_search_params={'cutoff':{'maxsteps':maxsteps_linesearch},'alpha':{'maxsteps':maxsteps_linesearch}}
        converged=False
        while self._it<maxsteps:
            for name in names:
                other_values={p:v for p,v in self.opt_param_values.items() if p!=name}
                line_search_params[name]['start']=self.opt_param_values[name]
                line_search_params[name]['inc']=incs[self._it]
                if self._it==0 and name=='cutoff':
                    evsteps=evo_steps0
                else:
                    evsteps=evo_steps
                opt_values,param_evolution,energies=self.cmera.optimizeParameter(cost_fun=cost_fun,cost_fun_params=cost_fun_params,
                                                                          name=name,
                                                                          delta=delta,
                                                                          evo_steps=evsteps,
                                                                          test_delta=test_delta,
                                                                          test_steps=test_steps,
                                                                          line_search_params=line_search_params[name],
                                                                          precision=precision,
                                                                          other_parameter_values=other_values,                                                                      
                                                                          pinv=pinv,
                                                                          tol=tol,
                                                                          Dthresh=Dthresh,
                                                                          trunc=trunc,
                                                                          Dinc=Dinc,
                                                                          ncv=ncv,
                                                                          numeig=numeig,
                                                                          thresh=thresh,
                                                                          maxsteps=optimizerSteps,
                                                                          plot=False)
                for n,v in param_evolution.items():
                    self.accumulated_parameter_values[n].extend(v)
                self.accumulated_energies.extend(energies)
                self._diffs[name]=np.abs(self.opt_param_values[name]-opt_values[name])
                self.opt_param_values[name]=opt_values[name]
                
                plt.figure(1,figsize=(12,3))
                # if name=='cutoff':
                #     index=1
                # elif name=='alpha':
                #     index=2
                ax=plt.subplot(1,3,1)
                ax.clear()
                plt.plot(self.accumulated_parameter_values['cutoff'])
                plt.legend(['cutoff'],fontsize=12,loc='best')
                
                ax=plt.subplot(1,3,2)                
                ax.clear()
                plt.plot(self.accumulated_parameter_values['alpha'])
                plt.legend(['alpha'],fontsize=12,loc='best')
                    
                ax=plt.subplot(1,3,3)
                ax.clear()
                plt.plot(self.accumulated_energies)
                plt.legend(['energy'],loc='best',fontsize=12)
        
                plt.draw()
                plt.show()
                plt.tight_layout()
                plt.pause(0.01)
                #cmera.canonize() to get the exact value, you need to canonize; however, to get a rough idea, it's enough to use the
                #last value of the lambdas
                print(f'at step {self._it}: S={-(self.cmera.lam**2).dot(np.log(self.cmera.lam**2))}, D={len(self.cmera.lam)}')
                
            
            if (eps>1E-16) and (np.max(list(self._diffs.values()))<eps):
                converged=True
                break
            if self._it%savestep==0:
                print('checkpointing optimizer to the file',self.name)
                self.save(self.name)
            self._it+=1
        return self.opt_param_values,self.accumulated_parameter_values,self.accumulated_energies,converged

    
    def optimizeGaussianEntangler(self,
                                  maxsteps=20,
                                  optimizerSteps=4,
                                  eps=1E-12,
                                  trunc=1E-10,
                                  tol=1E-12,
                                  incs=None,
                                  evo_steps=20,
                                  test_steps=6,
                                  delta=0.001j,
                                  test_delta=0.025j,
                                  maxsteps_linesearch=100,
                                  evo_steps0=60, 
                                  precision=0.001,
                                  pinv=1E-200,
                                  Dthresh=1E-6,
                                  Dinc=1,
                                  ncv=30,
                                  numeig=6,
                                  thresh=1E-8,
                                  savestep=10):
        
        self.optimizeEntangler(names=['cutoff','alpha'],
                               maxsteps=maxsteps,
                               eps=eps,
                               incs=incs,
                               maxsteps_linesearch=maxsteps_linesearch,
                               evo_steps0=evo_steps0,
                               savestep=savestep,
                               cost_fun=cmeralib.measure_energy_free_boson_with_cutoff,
                               cost_fun_params={'cutoff':1.0},
                               optimizerSteps=optimizerSteps,
                               trunc=trunc,
                               tol=tol,
                               evo_steps=evo_steps,
                               test_steps=test_steps,
                               delta=delta,
                               test_delta=test_delta,
                               precision=precision,
                               pinv=pinv,
                               Dthresh=Dthresh,
                               Dinc=Dinc,
                               ncv=ncv,
                               numeig=numeig,
                               thresh=thresh)

        


    

        

    
