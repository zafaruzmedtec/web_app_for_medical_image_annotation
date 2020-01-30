from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage.exposure import rescale_intensity
import cv2


def testGenerator(img,target_size = (256,256),flag_multi_class = False,as_gray = True):
    img = img/255.
    img = trans.resize(img,target_size)
    #img = rescale_intensity(img)
    img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    img = np.reshape(img,(1,)+img.shape)
    yield img


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def postProcess(npyfile,imgHeight,imgWidth,flag_multi_class = False,num_class = 2):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(1,1))
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        img = img * 255
        ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        resized_img_to_original_size = cv2.resize(thresh1,(int(imgWidth),int(imgHeight))) #imgHeight, imgWidth
        #opening = cv2.morphologyEx(resized_img_to_original_size, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(resized_img_to_original_size, cv2.MORPH_CLOSE, kernel) # remove small black points on the segmented contour 
        #img_name_ext = os.path.basename(img_names[i])
        #img_name = os.path.splitext(img_name_ext)[0]
        #io.imsave(os.path.join(save_path, img_name + '_predicted.png'),thresh1)
        #cv2.imwrite(save_path + img_name + '_predicted.png', thresh1)
        return closing.astype(np.uint8)