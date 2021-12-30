'''
Author: chen kai
Date: 2021-12-30
Description: Transform original DICOM images (size 256*256) to PNG images after zooming them with Lanczos-3 Interpolation Kernel.
Inputs: 
    Folder -- The path of the folder where original DICOM images arestored, such as "DICOMImages/ARGUILLOT HELENA/"
    SaveFolder -- The path of the folder where the output zoomed PNG images will be stored.
    zoomfactor -- Zoom factor. 
Outputs: 
    PNG images stored in the SaveFolder.

package installation: pydicom
'''

import pydicom
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from Transformer import ConvertDICOMDirToNumpyArray
import tensorflow as tf
import cv2 
import logging
file_dir = "DICOMImages"
zoom_factor = 4
logging.basicConfig(level=logging.DEBUG,
                    filename='ResizeDICOMImg_lanczos5.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )

save_dir = "Resize_factor_zoom_factor{}".format(zoom_factor)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for person_dir in os.listdir(file_dir):
    dicom_dir = os.path.join(file_dir, person_dir)
    patient_dir = os.path.join(save_dir, person_dir)
    if not os.path.exists(patient_dir):
        os.mkdir(patient_dir)
    img_name = os.listdir(dicom_dir)
    img_array = ConvertDICOMDirToNumpyArray(dicom_dir)
    if len(img_name) != len(img_array):
        logging.error(dicom_dir + "内的dicom文件部分有误，请查看(比如空dicom文件建议删除)")
        logging.error(dicom_dir + "未处理!")
        continue
    for index in range(len(img_name)):
        width, height = img_array[index].shape
        img_temp = img_array[index][np.newaxis,:,:, np.newaxis]
        img_resize = tf.image.resize(img_temp, (width*zoom_factor, height*zoom_factor), 'lanczos5')[0,...,0].numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(patient_dir, img_name[index] + ".png"), img_resize)

    logging.info(dicom_dir + "所有dicom文件处理完毕!")



