import numpy as np
def polar_grid(M):
    [x, y] = np.meshgrid(np.linspace(0, M-1, M) - M / 2,np.linspace(0, M-1, M)-M/2)
    r = np.sqrt(x**2+y**2)
    r = np.where(r < M-1, r , M-1)
    theta = np.arctan2(y,x)
    return r, theta

if __name__ == "__main__":
    #full test
    pass