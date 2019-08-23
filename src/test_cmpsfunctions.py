import src.cMERAlib as cmeralib
import numpy as np
import src.cMERAcmpsfunctions as cmf
def test_mixed_cmpo_transfer_operator():
    D=10
    δ = 0.005j
    β = 1.0
    Λ = 1
    γ = 0.8
    Q = np.random.rand(D,D)
    R = [np.random.rand(D,D)]
    Gammas = cmeralib.free_boson_entangler_propagator(α=1/4,Λ=Λ,δ=(1+1j)/2*δ)
    M = Gammas[0][0].shape[0]
    Q_=np.kron(np.eye(Q.shape[0]),Gammas[0][0])+np.kron(Q,np.eye(M))+np.kron(R[0],Gammas[0][1])
    R_=[np.kron(np.eye(R[0].shape[0]),Gammas[1][0])+np.kron(R[0],np.eye(M) + Gammas[1][1])]
    
    vector = np.reshape(np.eye(D*M),(D*M)**2)
    
    out = cmf.mixed_cmps_cmpo_transfer_operator(Q, R, Gammas,
                                      Q, R, Gammas,
                                      direction=1, vector=vector)
    out2 = cmf.transfer_operator(Q_, R_,direction=1,vector=vector)
    np.testing.assert_allclose(out, out2)

    out = cmf.mixed_cmps_cmpo_transfer_operator(Q, R, Gammas,
                                      Q, R, Gammas,
                                      direction=-1, vector=vector)
    out2 = cmf.transfer_operator(Q_, R_,direction=-1,vector=vector)
    np.testing.assert_allclose(out, out2)
    
