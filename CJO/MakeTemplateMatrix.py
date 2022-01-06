import numpy as np
def MakeTemplateMatrix(H):
    num_ele = np.prod(H[0][0].shape)
    num_H = len(H) * len(H[0])
    U = np.zeros((num_ele,num_H))

    for m in range(len(H)):
        for n in range(len(H[0])):
            U[:, m * len(H[0]) + n] = H[m][n].flatten('F')
    return U