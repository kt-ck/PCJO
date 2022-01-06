from ComplexSteerableAngularFunctions import ComplexSteerableAngularFunctions
from RaisedCosineRadialFunctions import RaisedCosineRadialFunctions
from polar_grid import polar_grid
import numpy as np
from Laguerre2D import Laguerre2D
import math
def GenerateOrientedChannels(M,K,J=1,angular_response='steerable',radial_response='LG',LGscaleParam=25,filter_type='symmetric'):
    if angular_response == "steerable":
        H_ang = ComplexSteerableAngularFunctions(M,K)
    elif angular_response == "none":
        K=1
        H_ang = [1]


    if radial_response == "raised-cosine":
        H_rad = RaisedCosineRadialFunctions(J,M);        
    elif radial_response == 'LG':
        H_rad = []
        r, theta = polar_grid(M)
        for j in range(J):
            H_rad.append(np.abs(np.fft.fftshift(np.fft.fft2(Laguerre2D(np.array([M,M]),j+1,LGscaleParam,r)))))
            H_rad[j][0,:] = 0
            H_rad[j][:,0] = 0            
    
    elif radial_response == 'scale-shiftable':
        rang = np.linspace(1 - (M+1)/2, M-(M+1)/2, M)

        [X,Y] = np.meshgrid(rang, rang)
        r = 2*math.pi/M*np.sqrt(X**2 + Y**2)
        sic = 3
        d = 2
        H_rad = []
        for j in range(J):
            H_rad.append(np.sinc((np.log2(r/math.pi+np.finfo(float).eps)+j)))
            H_rad[j] = np.where(r > math.pi, 0, H_rad[j])
      
    #cell(J,K)
    H = []
    for j in range(J):
        temp = []
        for k in range(K):  
            temp.append(np.fft.ifft2(np.fft.ifftshift(H_ang[k]*H_rad[j]))) 
            # H{j,k} = ifft2(ifftshift(H_ang{k}.*H_rad{j})); % Eq. (7.22)
            if angular_response == 'steerable':           
                if filter_type == 'symmetric':                   
                    temp[k] = np.fft.fftshift(np.real(temp[k]))
                else:
                    temp[k] = np.fft.fftshift(np.imag(temp[k]))
    
            if radial_response == 'raised-cosine':
                temp[k] = np.fft.fftshift(temp[k])
        H.append(temp)
    return H

if __name__ == "__main__":
    #full test
    ans = GenerateOrientedChannels(5,4,4,'steerable','scale-shiftable',math.sqrt(2*math.pi)*1,'symmetric')
    print(len(ans),len(ans[0]),ans[0][0].shape)
    print(ans)