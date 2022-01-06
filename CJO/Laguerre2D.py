import numpy as np
import math
def Laguerre1D(j,x):
    val = np.zeros(x.shape)
    for jp in range(j + 1):
        val = val + ((-1)**jp) * (np.math.factorial(j)/(np.math.factorial(jp)*np.math.factorial(j-jp)))*(x**jp)/(np.math.factorial(jp))
    return val

def Laguerre2D(dim,j,a,r=-1):
    if len(dim.shape) == 1:
        dim = np.array([dim[0],dim[0]])
     

    if r == -1:
        start = np.linspace(1-(dim[1] + 1)/2, dim[1] - (dim[1] + 1)/2, dim[1])
        end = np.linspace(1-(dim[0] + 1)/2, dim[0] - (dim[0] + 1)/2, dim[0])
        [X,Y] = np.meshgrid(start, end)
        r = np.sqrt(X**2 + Y**2)

    val = (1/np.sqrt(2*math.pi))*2*np.sqrt(math.pi)/a * np.exp(-(math.pi*r**2)/(a**2))*Laguerre1D(j,2*math.pi*r**2/(a**2))
    return val