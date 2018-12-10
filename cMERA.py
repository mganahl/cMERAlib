#!/usr/bin/env python
import numpy as np
import pickle
import os,sys,select
import datetime
import src.cMERAcmpsfunctions as cmf
import src.cMERAlib as cmeralib

import src.cMERAclass as cmera
from src.cMERAclass import cMERA #neccessary for unpickling
import src.utils as utils
import matplotlib.pyplot as plt
import math
import argparse
import warnings
import re
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))
    
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


def calculateCorrelator(data_accumulator,cmera,operators,N1=10,N2=40000,eps1=1E-4,eps2=4E-2):
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
    lamtens=cmera.lam    
    Qltens=cmera.Ql
    Rltens=cmera.Rl

    #if lamtens.shape[0]!=Qltens.shape[0]:
    #    lamtens,Qltens,Rltens,Qrtens,Rrtens,rest1=cmf.canonize(cmera.Ql,[cmera.Rl],linit=None,rinit=None,maxiter=100000,tol=tol,\
    #                                                           ncv=ncv,numeig=numeig,pinv=pinv,thresh=thresh,trunc=trunc,Dmax=cmera.D,verbosity=0)
        
    x=np.append(np.arange(1,N1+1)*eps1,np.arange(2,N2+1)*eps2)
    corr1,vec1=cmf.calculateRelativisticCorrelators(Ql=Qltens,Rl=Rltens,r=np.diag(lamtens**2),cutoff=cmera.cutoff,operators=operators,dx=eps1,N=N1,initial=None)
    corr2,vec2=cmf.calculateRelativisticCorrelators(Ql=Qltens,Rl=Rltens,r=np.diag(lamtens**2),cutoff=cmera.cutoff,operators=operators,dx=eps2,N=N2,initial=vec1)
    #=cmf.CorrCorr(Qltens,Rltens,np.diag(lamtens**2),eps1,N1,cmera.cutoff,initial=None)
    #=cmf.CorrCorr(Qltens,Rltens,np.diag(lamtens**2),eps2,N2,cmera.cutoff,initial=vec1)
    corr=np.append(corr1,corr2[1::])    
    label=operators[0]+'(0)'+operators[1]+'(x)'
    if  label not in data_accumulator:
        data_accumulator[label]=[x,(cmera.scale,corr)]
    else:
        data_accumulator[label].append((cmera.scale,corr))

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
    lamtens=cmera.lam    
    Qltens=cmera.Ql
    Rltens=cmera.Rl

    
    psi=np.trace(Rltens.dot(np.diag(lamtens)).dot(np.diag(lamtens)))
    
    if 'psi' not in data_accumulator:
        data_accumulator['psi']=[psi]
    else:
        data_accumulator['psi'].append(psi)        
    return data_accumulator

def calculateDensityObservables(data_accumulator,cmera):
    """
    calculates the observable <psidag psi> (particle density) using the cMPS tensors from cmera
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
    

    lamtens=cmera.lam
    Qltens=cmera.Ql
    Rltens=cmera.Rl    

    dens=np.trace(Rltens.dot(np.diag(lamtens)).dot(np.diag(lamtens)).dot(herm(Rltens)))
    
    if 'density' not in data_accumulator:
        data_accumulator['density']=[dens]
    else:
        data_accumulator['density'].append(dens)        
    return data_accumulator

def checkWicksTheorem(data_accumulator,cmera,N=20000,eps=0.01):
    lamtens=cmera.lam
    Qltens=cmera.Ql
    Rltens=cmera.Rl    
    
    x=np.arange(1,N+1)*eps
    wick=cmeralib.checkWickTheorem(Qltens,Rltens,np.diag(lamtens**2),eps,N)
    if 'wick_theorem' not in data_accumulator:
        data_accumulator['wick_theorem']=[x,wick]
    else:
        data_accumulator['wick_theorem'].append(wick)
    return data_accumulator



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
            plt.loglog(data_accumulator['pi(0)pi(x)'][0],np.abs(data_accumulator['pi(0)pi(x)'][-1][1]),'-b')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pi(0)pi(x)'][0],np.abs(data_accumulator['pi(0)pi(x)'][-1][1]),'-b')
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
            plt.loglog(data_accumulator['pi(0)pi(x)'][0],np.abs(data_accumulator['pi(0)pi(x)'][-1][1]),'-b',data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylim([1E-12,100])
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['pi(0)pi(x)'][0],np.abs(data_accumulator['pi(0)pi(x)'][-1][1]),'-b',data_accumulator['pipi_exact'][0],data_accumulator['pipi_exact'][-1],'--k')
            plt.ylabel(r'$|\langle\pi(x)\pi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.legend(['numerical evolution','exact free'],loc='best')    
            plt.tight_layout()
            plt.draw()
            plt.show()
            plt.pause(0.05)

        except KeyError:
            pass
        
    if ('dphidphi' in which) or ('dxphidxphi' in which):
        try:
            plt.figure(2)
            plt.clf()
            plt.subplot(2,1,1)
            plt.title(title)    
            plt.loglog(data_accumulator['dxphi(0)dxphi(x)'][0],np.abs(data_accumulator['dxphi(0)dxphi(x)'][-1][1]),'-b')
            plt.ylabel(r'$|\langle\partial\phi(x)\partial\phi(y)\rangle|$',fontsize=20)
            plt.xlabel(r'$x-y$',fontsize=20)
            plt.ylim([1E-12,100])
            plt.tight_layout()
            
            plt.subplot(2,1,2)
            plt.title(title)
            plt.semilogx(data_accumulator['dxphi(0)dxphi(x)'][0],np.abs(data_accumulator['dxphi(0)dxphi(x)'][-1][1]),'-b')    
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

    if ('density' in which) or ('n' in which):
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
    parser.add_argument('--info_cMERA', help='print a small manual',action='store_true')    
    parser.add_argument('--delta',help='imaginary part of the time step for unitary time evolution (0.001)',type=float,default=0.001)
    parser.add_argument('--Dmax', help='maximal cMPS bond dimension (32)',type=int,default=32)
    parser.add_argument('--nwarmup', help='number of initial warmup-steps without truncation (2); ',type=int,default=2)    
    parser.add_argument('--Dinc', help='bond dimension increment (1)',type=int,default=1)    
    parser.add_argument('--cutoff', help='UV cutoff of the entangler (1.0)',type=float,default=1.0)
    parser.add_argument('--alpha', help='entangling strength; if not given, alpha=cutoff/4 (None)',type=float,default=None)    
    parser.add_argument('--invrange', help='inverse interctionrange (1.0)',type=float,default=1.0)    
    parser.add_argument('--pinv',help='pseudoinver cutoff (1E-20); if chosen too large, severe artifacts will show up',type=float,default=1E-20)
    parser.add_argument('--operators', nargs='+',help="entangling operators; list of length 2 or 4 of str. \n for length 2: elements have to be  'n'; \n for length 4: use any of the following: ['pi','phi','psi','psidag']",type=str,default=['n','n'])
    parser.add_argument('--trunc',help='truncation threshold (1E-10); all schmidt-values below trunc will be discarded, irrespective of Dmax',type=float,default=1E-10)
    parser.add_argument('--Dthresh',help='truncation threshold at which the bond dimension is increased by Dinc (1E-6)',type=float,default=1E-6)    
    parser.add_argument('--thresh',help='threshold for "large-imaginary-eigenvalue" error (1E-10); dont worry about it',type=float,default=1E-10)        
    parser.add_argument('--tol', help='tolerance of eigensolver for finding left and right reduced DM (1E-10)',type=float,default=1E-10)
    parser.add_argument('--imax', help='maximum number of iterations (5000)',type=int,default=5000)
    parser.add_argument('--checkpoint', help='save the simulation every checkpoint iterations for checkpointing (100)',type=int,default=100)
    #parser.add_argument('--resume_checkpoint', help='load a checkpointed file and resume simulation',type=str)
    parser.add_argument('--filename', help='filename for output (_interactingBosoncMERA)',type=str,default='_interactingBosoncMERA')
    parser.add_argument('--noTruncAfterFree', help='do not truncate after free propagation (False)',action='store_true')
    parser.add_argument('--noTruncAfterInt', help='do not truncate after interacting propagation (False)',action='store_true')
    parser.add_argument('--loaddir', help='filename of the simulation to be loaded; the resumed simulation will be stored in filename (see above)',type=str)
    parser.add_argument('--parameterfile', help='read parameters from a given file; each line in the file has to contain the parameter name and its value seperated by a whitespace; values passed by file override values passed by command line',type=str)    
    #parser.add_argument('--ending', help='suffix of the file names: Ql+args.ending, Rl+args.ending, lam+args.ending ',type=str)
    parser.add_argument('--numeig', help='number of eigenvector in TMeigs (5)',type=int,default=5)
    parser.add_argument('--ncv', help='number of krylov vectors in TMeigs (40)',type=int,default=40)
    parser.add_argument('--show_plots', nargs='+',help='list of strings from {pipi,exact,dphidphi,density,psi,lams,tw}',type=str,default=[''])
    parser.add_argument('--measurestep', help='calculate observables ever measurestep; if 0, nothing is measured (0)',type=int,default=0)
    parser.add_argument('--measure', nargs='+',help='list of strings from {pipi,exact,dxphidxphi,dphidphi,density,n,psi,lams,tw}, where n=density and dphidphi=dxphidxphi',type=str,default=[''])    
    parser.add_argument('--inter', help='interaction (0.0)',type=float,default=0.0)
    parser.add_argument('--keepcp', help='keep old checkpoint files of the simulation',action='store_true')
    parser.add_argument('--N1', help='number of points for calculating correlators at distances np.arange(N1**eps1 (10)',type=int,default=10)
    parser.add_argument('--N2', help='number of points for calculating correlators at distances eps1*N1+np.arange(N2)*eps2 (40000)',type=int,default=40000)
    parser.add_argument('--N3', help='number of points for calculating violation of wicks theorem at distances eps3*N3 (20000)',type=int,default=20000)        
    parser.add_argument('--eps1', help='discretization for calculating correlators',type=float,default=1E-4)
    parser.add_argument('--eps2', help='discretization for calculating correlators',type=float,default=1E-2)
    parser.add_argument('--eps3', help='discretization for calculating violation of wicks theorem',type=float,default=1E-2)        

    args=parser.parse_args()

    if args.info_cMERA:
        help(cmera.cMERA)
        sys.exit()
    observables=['pipi','dxphidxphi','dphidphi','lam','density','n','psi','tw','wick','exact','']
    if args.parameterfile!=None:
        parameters=utils.read_parameters(args.parameterfile)
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
    init_params=dict(cutoff=args.cutoff,
                     alpha=args.alpha,
                     inter=args.inter,
                     invrange=args.invrange,
                     operators=args.operators,
                     delta=args.delta*1.0j,
                     nwarmup=args.nwarmup,
                     Dmax=args.Dmax,
                     dtype=complex)
    
    #store all evolution parameters in a dict()
    evolution_params=dict(cutoff=args.cutoff,
                          alpha=args.alpha,
                          inter=args.inter,
                          invrange=args.invrange,
                          operators=args.operators,
                          delta=args.delta*1.0j,
                          truncAfterFree=not args.noTruncAfterFree,
                          truncAfterInt=not args.noTruncAfterInt,                          
                          pinv=args.pinv,
                          tol=args.tol,
                          Dthresh=args.Dthresh,
                          trunc=args.trunc,
                          Dinc=args.Dinc,
                          ncv=args.ncv,
                          numeig=args.numeig,
                          thresh=args.thresh)
    
    cmera_sim=cmera.cMERA(**init_params)
    if args.loaddir!=None:
        try:
            cmera_sim.load(root+'/'+args.loaddir)
        except FileNotFoundError:
            cmera_sim.load(args.loaddir)            


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
                data_accumulator=calculateCorrelator(data_accumulator,cmera_sim,operators=['pi','pi'],N1=args.N1,eps1=args.eps1,N2=args.N2,eps2=args.eps2)
            if ('dphidphi' in args.measure) or ('dxphidxphi' in args.measure):
                data_accumulator=calculateCorrelator(data_accumulator,cmera_sim,operators=['dxphi','dxphi'],N1=args.N1,eps1=args.eps1,N2=args.N2,eps2=args.eps2)                
                #data_accumulator=calculatedPhidPhiCorrelators(data_accumulator,cmera_sim)
            if ('exact' in args.measure):
                data_accumulator=calculateExactCorrelators(data_accumulator,cmera_sim.scale,args.cutoff)
            if 'psi' in args.measure:
                data_accumulator=calculatePsi(data_accumulator,cmera_sim)
            if ('density' in args.measure) or ('n' in args.measure):
                data_accumulator=calculateDensityObservables(data_accumulator,cmera_sim)
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
                
        sys.stdout.write('\r iteration %i, D=%i Dmax=%i, tw %.10f' %(cmera_sim.iteration,len(cmera_sim._lam),cmera_sim.Dmax,cmera_sim.truncated_weight))
        sys.stdout.flush()

