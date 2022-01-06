import numpy as np
import math
def GaussianSignal(dim,sigma,alpha,pos=np.array([-1, -1]),theta=None):
    '''
    -Generate an image of a Gaussian of dimensions dim,
    with a standard deviation of sigma and magnitude of
    alpha.
    -If pos is not specified, then the Gaussian is centered
    in the image.  pos is specified by (x,y) not (i,j)
    '''

    if len(dim.shape) == 1:
        dim = np.array([dim[0], dim[0]])

    if np.any(pos == -1):
        pos = (dim - 1) / 2
        pos = pos[::-1]

    x = np.linspace(0, dim[1]-1, dim[1])
    y = np.linspace(0, dim[0]-1, dim[0])
    [X,Y] = np.meshgrid(x,y)

    if theta == None:
        img = alpha* np.exp(-(1/(2*sigma**2))*((X-pos[0])**2 + (Y-pos[1])**2))
    else:
        rotM = np.array([[math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)]])
        X = (X - pos[0])
        Y = (Y - pos[1])
        X2 = (rotM[0,0]*X + rotM[0,1]*Y)/sigma[0]
        Y2 = (rotM[1,0]*X + rotM[1,1]*Y)/sigma[1]
        img = alpha*np.exp(-0.5*(X2**2+Y2**2))

    return img

if __name__ == "__main__":
    #full test
    ans = GaussianSignal(np.array([65]),np.array([3,1.5]),0.3,np.floor(np.array([32.5,32.5])),math.pi/4)
    print(ans.shape)
    print(ans)
