import numpy as np
import math
def SteeringMtx(K=4,theta=1):
    G = np.zeros((K,K))
    for m in range(K):
        for n in range(K):
            alpha = math.pi / K * (m - n + K + theta)
            if abs(alpha) < 1e-12:
                G[m,n] = 1
            else:
                G[m,n] = np.sin(K*alpha) / np.sin(alpha)/K
    return G

if __name__ == "__main__":
    print(SteeringMtx(3,2))