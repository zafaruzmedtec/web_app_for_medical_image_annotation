import numpy as np 
import os
import glob
import skimage.io as io
import sklearn
from sklearn.preprocessing import MinMaxScaler
import cv2
from skimage.transform import resize, rescale
from skimage.exposure import rescale_intensity
import pydicom
from skimage.draw import polygon
import plistlib
import skimage.transform as trans

image_path = '../INbreast/images/'
label_path = '../INbreast/groundtruth/'

image_name_path_arr = glob.glob(os.path.join(image_path,"*.tif"))
label_name_path_arr = glob.glob(os.path.join(label_path,"*.tif"))

img_name = []
label_name = []
for i in range(len(image_name_path_arr)):
    img_name_ext = os.path.basename(image_name_path_arr[i])
    img_name.append(os.path.splitext(img_name_ext)[0])
for i in range(len(label_name_path_arr)):
    label_name_ext = os.path.basename(label_name_path_arr[i])
    label_name.append(os.path.splitext(label_name_ext)[0])

def display(image, display_min, display_max): 
    image = np.array(image, copy=True)
    image.clip(display_min, display_max, out=image)
    image -= display_min
    np.floor_divide(image, (display_max - display_min + 1) / 256, out=image, casting='unsafe')
    return image.astype(np.uint8)

def lut_display(image, display_min, display_max) :
    lut = np.arange(2**16, dtype='uint16')
    lut = display(lut, display_min, display_max)
    return np.take(lut, image)
	
train_img_folder = './data/inbreast/train/image/'
train_label_folder = './data/inbreast/train/label/'
test_img_folder = './data/inbreast/test/image/'
test_label_folder = './data/inbreast/test/label/'

for i in range(len(label_name_arr)):
    img16Bit = io.imread(image_path + label_name[i] + '.tif')
    label = io.imread(label_name_arr[i])
    minIntensity = np.min(img16Bit)
    maxIntensity = np.max(img16Bit)

    img8Bit = lut_display(img16Bit, minIntensity, maxIntensity)
    print(np.min(img8Bit))
    print(np.max(img8Bit))
    
    if img8Bit.shape == (4084, 3328):
        if np.sum(img8Bit[0:-1, 0:50]) > 0:
            img8Bit = np.delete(img8Bit, np.s_[3142:-1], 1)
            label = np.delete(label, np.s_[3142:-1], 1)
            img8Bit = cv2.resize(img8Bit,(int(3328),int(2560)))
            #print(np.min(img8Bit))
            #print(np.max(img8Bit))
        else:
            img8Bit = np.delete(img8Bit, np.s_[0:186], 1)
            label = np.delete(label, np.s_[0:186], 1)
            img8Bit = cv2.resize(img8Bit,(int(3328),int(2560)))
            #print(np.min(img8Bit))
            #print(np.max(img8Bit))
    
	new_size = (256, 256)
    img8Bit = cv2.resize(img8Bit,new_size)
    label = cv2.resize(label,new_size)
    #print('Resize 256: ', np.min(img8Bit))
    #print('Resize 256: ', np.max(img8Bit))
    img8Bit = rescale_intensity(img8Bit)
    #print('Rescale 0-255: ', np.min(img8Bit))
    #print('Rescale 0-255: ', np.max(img8Bit))
    
    # split into Train 80% (85 img), Test 20% (22 img)
    if i < 22:
        cv2.imwrite(test_img_folder + label_name[i] + '.png', img8Bit)
        cv2.imwrite(test_label_folder + label_name[i] + '.png', label)
    else:
        cv2.imwrite(train_img_folder + label_name[i] + '.png', img8Bit)
        cv2.imwrite(train_label_folder + label_name[i] + '.png', label)

