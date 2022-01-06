from SteeringMtx import SteeringMtx
import math
import scipy
import scipy.linalg
import numpy as np
def CHotellingAsymmetricTraining(IS,X_alpha, IN,Temps,num_orient,num_scales,Technique,thetaS, scaleS, ampS):
    # print(Temps.shape, IS.shape, X_alpha.shape)
    vS = Temps.T @ IS
    vN = Temps.T @ IN
    vX = Temps.T @ X_alpha

    NImg=IS.shape[1]
    x0prime = []
    if Technique == 'jde':
        for i in range(NImg):
            RO = SteeringMtx(num_orient,thetaS[i] / math.pi*num_orient)
            RS = scipy.linalg.toeplitz(np.sinc((np.linspace(0, 1-num_scales, num_scales)-scaleS[i])),
                                      np.sinc((np.linspace(0, num_scales-1, num_scales)-scaleS[i])))
            R = np.kron(RS,RO)
            res = 1/ampS[i]*R.T@vX[:, i]
            x0prime.append(res) 
    x0prime = np.array(x0prime).T
    vBar = np.mean(x0prime,axis=1,keepdims=True)
    
    S = 0.5 * np.cov(vN) + 0.5*np.cov(vS-vX)
    condS = np.linalg.cond(S)
    
    D, V = scipy.linalg.eig(S)

    D = np.real(D)
    Diag_vec = D/(D**2+(1e-14)*np.max(D)**2)
    S_inv = V*np.diag(Diag_vec)*V.T

    wCh = S_inv @ vBar; 

    NormN = Temps.flatten('F')@Temps.flatten('F').T

    T = np.zeros((num_scales, num_scales))
    for m in range(num_scales):
        for n in range(num_scales):
            T1 = Temps[:, m*num_orient:m*num_orient + num_orient]
            T2 = Temps[:, n*num_orient:n*num_orient + num_orient] 
            T[m,n]= T1.flatten('F') @ T2.flatten('F').T
    tS, tN = [], []
    for j in range(NImg):
        RSO = SteeringMtx(num_orient,thetaS[j]/math.pi*num_orient)
        RSS = scipy.linalg.toeplitz(np.sinc(np.linspace(0, 1-num_scales, num_scales)-scaleS[j]),
            np.sinc(np.linspace(0, num_scales-1, num_scales)-scaleS[j]))
        R = RSS.T @ RSS
        RS = np.kron(RSS,RSO)

        tS.append((ampS[j]**2 * (wCh.T @ (RS.T @ vS[:,j][:,np.newaxis] - vBar/2)/(R.flatten('F')@T.flatten('F').T)))[0,0])
        tN.append( (wCh.T @ (vN[:, j][:,np.newaxis]-vBar/2) / NormN)[0,0])

    tS = np.array(tS)
    tN = np.array(tN)
    return tS,tN,wCh,vBar,condS

if __name__ == "__main__":
    pass
