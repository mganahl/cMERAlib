import sys,os
#root=os.getcwd()
#os.chdir('../')
#sys.path.append(os.getcwd())#add parent directory to path
#os.chdir(root)
import unittest
import numpy as np
import scipy as sp
import random
import math
import datetime as dt
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import src.cMERAcmpsfunctions as cmf
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

plt.ion()
class TestTMeigs(unittest.TestCase):
    def setUp(self):
        self.D=32
        self.d=3
        self.places=10
        def TMeigs_helper(direction,dtype):
            if dtype==float:
                self.Q=np.random.rand(self.D,self.D)-0.5
                self.R=[np.random.rand(self.D,self.D)-0.5 for d in range(self.d)]
            elif dtype==complex:
                self.Q=np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5)
                self.R=[np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5) for d in range(self.d)]
            return cmf.TMeigs(self.Q,self.R,direction=direction,numeig=6,init=None,maxiter=6000,tol=1e-10,ncv=40,which='LR')
        self.TMeigsHelper=TMeigs_helper
    def test_TMeigs_left_complex(self):
        eta,v=self.TMeigsHelper('left',complex)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Q,self.R,'l',v)-eta*v),0.0,places=self.places,msg="TMeigs_left_complex: v is a bad eigenvector")
    def test_TMeigs_right_complex(self):
        eta,v=self.TMeigsHelper('right',complex)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Q,self.R,'r',v)-eta*v),0.0,places=self.places,msg="TMeigs_left_complex: v is a bad eigenvector")        
    def test_TMeigs_left_float(self):
        eta,v=self.TMeigsHelper('left',float)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Q,self.R,'l',v)-eta*v),0.0,places=self.places,msg="TMeigs_left_complex: v is a bad eigenvector")
    def test_TMeigs_right_float(self):
        eta,v=self.TMeigsHelper('right',float)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Q,self.R,'r',v)-eta*v),0.0,places=self.places,msg="TMeigs_left_complex: v is a bad eigenvector")

class TestRegauge(unittest.TestCase):
    def setUp(self):
        self.D=32
        self.d=3
        self.places=10
        def regauge_helper(gauge,dtype):
            if dtype==float:
                self.Q=np.random.rand(self.D,self.D)-0.5
                self.R=[np.random.rand(self.D,self.D)-0.5 for d in range(self.d)]
                self.ih=np.random.rand(self.D,self.D)-0.5
            elif dtype==complex:
                self.Q=np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5)
                self.R=[np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5) for d in range(self.d)]
                self.ih=np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5)
                
            return cmf.regauge(self.Q,self.R,gauge=gauge,init=None,maxiter=100000,tol=1E-10,ncv=100,numeig=6,pinv=1E-200,thresh=1E-10)
        self.regaugeHelper=regauge_helper
        
    def test_regauge_left_complex(self):
        self.left,self.y,self.Ql,self.Rl=self.regaugeHelper('left',complex)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Ql,self.Rl,'l',np.eye(self.D))),0.0,places=self.places,msg="regauge_left_complex: regauging failed")
    def test_regauge_left_float(self):
        self.left,self.y,self.Ql,self.Rl=self.regaugeHelper('left',float)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Ql,self.Rl,'l',np.eye(self.D))),0.0,places=self.places,msg="regauge_left_float: regauging failed")
    def test_regauge_right_complex(self):
        self.right,self.x,self.Qr,self.Rr=self.regaugeHelper('right',complex)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Qr,self.Rr,'r',np.eye(self.D))),0.0,places=self.places,msg="regauge_right_complex: regauging failed")
    def test_regauge_right_float(self):
        self.right,self.x,self.Qr,self.Rr=self.regaugeHelper('right',float)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Qr,self.Rr,'r',np.eye(self.D))),0.0,places=self.places,msg="regauge_right_float: regauging failed")

class TestCanonize(unittest.TestCase):
    def setUp(self):
        self.D=32
        self.d=3
        self.places=10
        def regauge_helper(dtype):
            if dtype==float:
                self.Q=np.random.rand(self.D,self.D)-0.5
                self.R=[np.random.rand(self.D,self.D)-0.5 for d in range(self.d)]
                self.ih=np.random.rand(self.D,self.D)-0.5
            elif dtype==complex:
                self.Q=np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5)
                self.R=[np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5) for d in range(self.d)]
                self.ih=np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5)
                
            return cmf.canonize(self.Q,self.R,linit=None,rinit=None,maxiter=100000,tol=1E-10,ncv=100,numeig=6,pinv=1E-200,thresh=1E-10,trunc=1E-16,Dmax=self.D,verbosity=0)
        self.regaugeHelper=regauge_helper

    def test_canonize_complex(self):
        self.lam,self.Ql,self.Rl,self.Qr,self.Rr,rest=self.regaugeHelper(complex)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Ql,self.Rl,'l',np.eye(self.D))),0.0,places=self.places,msg="canonize_complex: regauging failed")
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Qr,self.Rr,'r',np.eye(self.D))),0.0,places=self.places,msg="canonize_complex: regauging failed")
        self.assertAlmostEqual(np.linalg.norm(self.lam),1.0,places=12,msg="canonize_complex: regauging failed")
        
    def test_canonize_float(self):
        self.lam,self.Ql,self.Rl,self.Qr,self.Rr,rest=self.regaugeHelper(float)
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Ql,self.Rl,'l',np.eye(self.D))),0.0,places=self.places,msg="canonize_float: regauging failed")
        self.assertAlmostEqual(np.linalg.norm(cmf.transferOperator(self.Qr,self.Rr,'r',np.eye(self.D))),0.0,places=self.places,msg="canonize_complex: regauging failed")                
        self.assertAlmostEqual(np.linalg.norm(self.lam),1.0,places=12,msg="canonize_complex: regauging failed")                
        
        
class TestLGMRES(unittest.TestCase):
    def setUp(self):
        #initialize a CMPS by loading it from a file 
        self.D=32
        self.d=3
        self.places=10
        def lgmres_helper(direction,dtype):
            if dtype==float:
                self.Q=np.random.rand(self.D,self.D)-0.5
                self.R=[np.random.rand(self.D,self.D)-0.5 for d in range(self.d)]
                self.ih=np.random.rand(self.D,self.D)-0.5
            elif dtype==complex:
                self.Q=np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5)
                self.R=[np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5) for d in range(self.d)]
                self.ih=np.random.rand(self.D,self.D)-0.5+1j*(np.random.rand(self.D,self.D)-0.5)
            self.dens,self.gauge,self.Q,self.R=cmf.regauge(self.Q,self.R,init=None,maxiter=100000,tol=1E-10,ncv=100,numeig=6,pinv=1E-200,thresh=1E-10)
            if direction in (1,'left','l'):
                return cmf.inverseTransferOperator(Q=self.Q,R=self.R,l=np.eye(self.D),r=self.dens,ih=self.ih,direction=direction,x0=None,tol=1e-12,maxiter=4000)
            elif direction in (-1,'right','r'):
                return cmf.inverseTransferOperator(Q=self.Q,R=self.R,r=np.eye(self.D),l=self.dens,ih=self.ih,direction=direction,x0=None,tol=1e-12,maxiter=4000)
        self.lgmresHelper=lgmres_helper
        
    def test_lgmres_left_complex(self):
        v=self.lgmresHelper('left',complex)
        self.assertAlmostEqual(np.linalg.norm(self.ih-cmf.pseudotransferOperator(self.Q,self.R,np.eye(self.D),self.dens,'left',v)),0.0,places=self.places,msg="lgmres_left_complex: no accurate solutoin found")
    def test_lgmres_left_float(self):
        v=self.lgmresHelper('left',float)
        self.assertAlmostEqual(np.linalg.norm(self.ih-cmf.pseudotransferOperator(self.Q,self.R,np.eye(self.D),self.dens,'left',v)),0.0,places=self.places,msg="lgmres_left_float: no accurate solutoin found")                
    def test_lgmres_right_complex(self):
        v=self.lgmresHelper('right',complex)
        self.assertAlmostEqual(np.linalg.norm(self.ih-cmf.pseudotransferOperator(self.Q,self.R,self.dens,np.eye(self.D),'right',v)),0.0,places=self.places,msg="lgmres_right_complex: no accurate solutoin found")                
    def test_lgmres_right_float(self):
        v=self.lgmresHelper('right',float)
        self.assertAlmostEqual(np.linalg.norm(self.ih-cmf.pseudotransferOperator(self.Q,self.R,self.dens,np.eye(self.D),'right',v)),0.0,places=self.places,msg="lgmres_right_float: no accurate solutoin found")                

if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestTMeigs)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestRegauge)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestCanonize)    
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestLGMRES)        
    unittest.TextTestRunner(verbosity=2).run(suite1)
    unittest.TextTestRunner(verbosity=2).run(suite2)
    unittest.TextTestRunner(verbosity=2).run(suite3)    
    unittest.TextTestRunner(verbosity=2).run(suite4)    
