import numpy as np
import math
def MVNLumpy(lump_width,muimg,sigimg,numimgs=1):
    nx, ny = muimg.shape
    dim = np.array([nx, ny])
    x = np.linspace(0, nx-1, nx)
    y = np.linspace(0, ny-1,ny)
    [X,Y] = np.meshgrid(x,y)

    center = (dim - 1) / 2
    kernel_width = lump_width / math.sqrt(2)

    kernel =  np.exp(-(1/(2*kernel_width**2))*((X-center[1])**2 + (Y-center[0])**2))
    
    kernel = kernel / np.sqrt(np.sum(kernel*kernel))
    
    kernel[round(kernel.shape[0]/2),round(kernel.shape[1]/2)] += 1e-2

    kernel_flatten = kernel.flatten('F')

    kernel = kernel / np.sqrt(kernel_flatten @ kernel_flatten.T)

    if numimgs == 1:
        n = np.random.randn(nx, ny)  
        img = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(n)) * np.fft.fft2(np.fft.fftshift(kernel)))))
        
   
        img = img * sigimg
    
        img = img + muimg
    else:
        img = np.zeros((np.prod(dim),numimgs))
        for i in range(numimgs):
            n = np.random.randn(nx, ny)
            # n = np.array([[-0.7540,-0.0534,1.0431],[-0.9239,0.1846,0.1569],[0.4161,1.1852,0.2770]])
            im = np.real(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(n)) * np.fft.fft2(np.fft.fftshift(kernel)))))
            im = im * sigimg
            im = im + muimg
            img[:, i] = im.flatten('F')
    return img
    
if __name__ == '__main__':
    ## full test
    ans = MVNLumpy(10, np.zeros((65,65)), np.ones((65,65)),500)
    print(ans.shape)
    print(ans)
    