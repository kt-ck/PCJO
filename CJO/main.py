import math
import numpy as np
from MVNLumpy import MVNLumpy
import cv2 
from GaussianSignal import GaussianSignal
from GenerateOrientedChannels import GenerateOrientedChannels
from MakeTemplateMatrix import MakeTemplateMatrix
from CHotellingAsymmetricTraining import CHotellingAsymmetricTraining
import matplotlib.pyplot as plt 
from WilcoxonAUC import WilcoxonAUC
import os


def Main(analysis_num_orient, analysis_num_scales):
    test_numimg=100
    training_numimg=500

    simulation = 0
    fixe_scale = 0
    fixe_orientation = 0     
    fixe_amplitude = 0        

    M = 65         

    training_lump_bk_param = 10    
    training_signal_sigma = np.array([2,1])

    amp_min = 0.01
    amp_max = 10


    training_signal_amp = amp_min+(amp_max-amp_min)*np.random.rand(training_numimg,1)
    if fixe_amplitude != 0:
        training_signal_amp = math.sqrt(amp_min*amp_max)*np.ones(training_numimg,1)

    training_signal_theta = np.random.rand(training_numimg,1)*math.pi

    if fixe_orientation != 0:
        training_signal_theta = math.pi / 4 * np.ones(training_numimg,1)

    scale_min= 1 
    scale_max= 2 


    training_signal_scale = scale_min+np.random.rand(training_numimg,1)*(scale_max-scale_min)

    if fixe_scale != 0:
        training_signal_scale = math.sqrt(scale_min*scale_max)*np.ones(training_numimg,1)

    WM_index = np.array([])
    test_lump_bk_param = training_lump_bk_param
    test_signal_sigma = training_signal_sigma

    test_signal_scale = scale_min+np.random.rand(test_numimg,1)*(scale_max-scale_min)
    if fixe_scale != 0:
        test_signal_scale = math.sqrt(scale_min*scale_max)*np.ones(test_numimg,1)


    test_signal_amp = amp_min+(amp_max-amp_min)*np.random.rand(test_numimg,1)
    if fixe_amplitude != 0:
        test_signal_amp =  math.sqrt(amp_min*amp_max)*np.ones(test_numimg,1)

    test_signal_theta = np.random.rand(test_numimg,1)*math.pi
    if fixe_orientation != 0:
        test_signal_theta = math.pi / 4 * np.ones(test_numimg,1)   

    if simulation != 0 :
        n = MVNLumpy(training_lump_bk_param,np.zeros((M, M)),np.ones((M, M)),training_numimg)
        s = MVNLumpy(training_lump_bk_param,np.zeros((M, M)),np.ones((M, M)),training_numimg)
    else:
        n, s = np.zeros((M*M, training_numimg)), np.zeros((M*M, training_numimg))
        for i in range(training_numimg):
            file = os.path.join("WM_regions","{}.png".format(i))
            img = cv2.imread(file)
            print(img)
            n[:,i] = np.reshape(cv2.imread(file), (M*M,1)).astype(np.float)
        
            file = os.path.join("WM_regions","{}.png".format(2*i))
            s[:, i] = np.reshape(cv2.imread(file), (M*M,1)).astype(np.float)

    x_alpha = np.zeros((M*M, training_numimg))
    for k in range(training_numimg):
        sig = GaussianSignal(np.array([M]),training_signal_sigma*training_signal_scale[k],training_signal_amp[k],
                            np.floor(np.array([M/2, M/2])), training_signal_theta[k])
        x_alpha[:, k] = sig.flatten('F') 
        s[:, k] += x_alpha[:, k]

    H = GenerateOrientedChannels(M,analysis_num_orient,analysis_num_scales,'steerable','scale-shiftable',math.sqrt(2*math.pi)*scale_min,'symmetric')
    U = MakeTemplateMatrix(H)
    tS,tN,wCh,x0, _ = CHotellingAsymmetricTraining(s,x_alpha, n,U,
                    analysis_num_orient,analysis_num_scales, 'jde',training_signal_theta, -np.log2(training_signal_scale), training_signal_amp)
    AUC_Training,tpf_Training,fpf_Training=WilcoxonAUC(tS,tN)
    
    print('AUC training={}SNR={} dB\n'.format(AUC_Training,"None"))
    print('AUC training={}'.format(AUC_Training))

    if AUC_Training<=0.60:
        print("Training performance is very poor - please adjust the parameters!")
    
    plt.plot(fpf_Training,tpf_Training)
    plt.show()
    ## test phase
#     if simulation != 0:
#         n = MVNLumpy(test_lump_bk_param,np.zeros((M,M)),np.ones(M),test_numimg)
#         s = MVNLumpy(test_lump_bk_param,np.zeros((M,M)),np.ones(M),test_numimg)
#     for k in range(test_numimg):
#         sig = GaussianSignal(M,test_signal_sigma*test_signal_scale[k],test_signal_amp[k],np.floor(np.array([M/2,M/2])),test_signal_theta[k])
#         s(:,k)=s(:,k)+sig(:);
    
#     else
#         % use the WMB backgrounds
#         for i=1:test_numimg,
#             file = ['WM_regions/' int2str(training_numimg+2*i-1) '.png'];
#             n(:,i) = double(reshape(imread(file), M*M,1));
#             %n(:,i) = double(reshape(rgb2gray(imread(file)), M*M,1));
#             file = ['WM_regions/' int2str(training_numimg+2*i) '.png'];
#             s(:,i) = double(reshape(imread(file), M*M,1));
#             %s(:,i) = double(reshape(rgb2gray(imread(file)), M*M,1)); 
#         end
        
#         for k=1:test_numimg,
#             sig = GaussianSignal(M,test_signal_sigma*test_signal_scale(k),test_signal_amp(k),floor([M/2 M/2]),test_signal_theta(k));
#             s(:,k)=s(:,k)+sig(:);
#         end   
#     end


#  [tS,tN,thetaS, ampS, scaleS]=CHotellingAsymmetricTesting(s,n,U,...
#                     analysis_num_orient,analysis_num_scales,'jde',wCh, [amp_min amp_max], [scale_min scale_max], x0);


# % The tS and tN are decision variable outputs which we can
# % perform ROC analysis on.

# [AUC_test,tpf_test,fpf_test]=WilcoxonAUC(tS,tN);
# AUC_Var_test = AUC_Var(tN, tS);

# ResultsFile=strcat('Results/CGB_test100training500amp001-10sigma51scale12M65_CGBsigma10/', 'K', int2str(analysis_num_orient), 'J', int2str(analysis_num_scales), '.mat');
# save(ResultsFile,'AUC_test','AUC_Var_test','tpf_test','fpf_test','test_signal_amp', 'ampS', 'test_signal_theta', 'thetaS', 'test_signal_scale','scaleS');
        
if __name__ == "__main__":
    print(os.getcwd())
    Main(5,3)

