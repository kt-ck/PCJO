from polar_grid import polar_grid
import numpy as np
import math
import scipy.interpolate
def radial_filter_response(N,scale):
    r = np.linspace(0,scale,int(N)).T
    lp = np.where(r<1/4, 1, 0) + np.where(np.logical_and(r<1/2, r>=1/4), 1, 0)*np.cos(math.pi/2*np.log2(np.finfo(float).eps+4*r))

    hp=np.where(np.logical_and(r<1/2,r>=1/4), 1, 0)* np.cos(math.pi/ 2*np.log2(np.finfo(float).eps+2*r)) + np.where(r>=1/2, 1, 0)

    lp = np.array([1] + lp.tolist() + np.conj(lp[::-1][1:]).tolist())
    hp = np.array([0] + hp.tolist() + np.conj(hp[::-1][1:]).tolist())
    return lp, hp

def radial_filters(J,M):
    lp, hp=radial_filter_response(M/2,1);  
    lp0 = lp
    # fb = cell(J+1,1)
    fb = []
    fb.append(hp)
    subsf=[1]
    for j in range(2, J + 1):
        lp, hp=radial_filter_response(M/2,2**(j-1))
        hp = hp * lp0
        lp0 = lp
        fb.append(hp)
        subsf.append(subsf[j-2]*2)

    if J == 0:
        fb[1] = 1
    else: 
        fb.append(lp)
        subsf.append(2**(J-1))
    return fb,subsf 



def RaisedCosineRadialFunctions(J,M):
    r, theta = polar_grid(M)
    [fb, subsf] = radial_filters(J,4*M)

    H=[]
    for j in range(0, J+1):
        print(fb[j])
        f = scipy.interpolate.interp1d(np.linspace(0,4*M-1, 4*M),fb[j], 'nearest')
        H.append(f(4 * r))
        if M % 2 == 0:
            H[j][0, :] = 0
            H[j][:, 0] = 0        
    return H



if __name__ == "__main__":
    #full test
    print(RaisedCosineRadialFunctions(2, 3))