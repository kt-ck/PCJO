import numpy as np
import math
from polar_grid import polar_grid

def ComplexSteerableAngularFunctions(M,K):
    r, theta=polar_grid(M)
    norm_const=math.factorial(K-1)/math.sqrt(K*math.factorial(2*K-2))

    H = []
    if K==1:
        H.append(np.ones((M,M)))
    else:
        for k in range(K):
            H.append((np.cos((theta-k*math.pi/K)))**(K-1)*norm_const*(
                np.where(np.abs(theta-k*math.pi/K) > math.pi/2, 0, 1) *
                np.where(abs(2*math.pi+theta-k*math.pi/K)>math.pi/2, 1, 0)*
                np.where(abs(-2*math.pi+theta-k*math.pi/K)>math.pi/2, 1, 0)))

            if M % 2 == 0:
                H[k][0,:]=0
                H[k][:,0]=0        
    return H

if __name__ == "__main__":
    polar_grid(3)
    print(ComplexSteerableAngularFunctions(4,3))
