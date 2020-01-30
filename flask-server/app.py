from flask import Flask, jsonify, request, render_template, redirect, Response
import random, json, urllib.request
import numpy as np
import cv2
import csv
#import skimage.io as io
from model import *
from data import *
from helper import *
from pandas import DataFrame
import json
from pymongo import MongoClient

app = Flask(__name__)

''' 
This is Segmentation REST API that requests image data from client side 
and perform the segmentation using pre-trained U-Net deep learning model,
contours of segmented image are sent back for visualising in the client side 
'''

@app.route('/segmentation', methods = ['POST'])
def segmentation():
    # Request and get json data
    req_data = request.get_json()
    print('Data received!')
    
    # Get the image array, height and width of the image from json data
    imgHeight = (req_data.get("height", "none"))
    imgWidth = (req_data.get("width", "none"))
    imgArray = (req_data.get("imgArray", "none"))
    #print('Image Height: ' + str(imgHeight) + ', Image Width: ' + str(imgWidth))
    
    # Build 16 bit image from image array
    img16Bit = np.reshape(imgArray, (imgHeight, imgWidth))
    print('Image Size: ' + str(img16Bit.shape))
    
    # Get min and max intensity of the image
    minIntensity = np.min(img16Bit)
    maxIntensity = np.max(img16Bit)
    print('Min Intensity: ' + str(minIntensity) + ', Max Intensity: ' + str(maxIntensity))
    
    # Convert 16 bit image to 8 bit and rescale between 0-255
    img8Bit = lut_display(img16Bit, minIntensity, maxIntensity)
    print('Image rescaled (0-255) and converted from 16 bit to 8 bit!')
    print('Min Intensity: ' + str(np.min(img8Bit)) + ', Max Intensity: ' + str(np.max(img8Bit)))
    
    # Run the pre-trained U-Net deep learning model
    testGene = testGenerator(img8Bit) # generate test image for model
    model = unet() # load network
    model.load_weights("unet_inbreast_85img.hdf5") # load model
    num_img = 1 # test image number
    results = model.predict_generator(testGene,num_img,verbose=1) # predict the mass lesion
    print('Prediction done!')
    
    # After prediction clear session of keras model
    K.clear_session()
    
    # Perform post processing (visualize predicted labels, threshold and remove small black holes inside the segmented contour)
    segImage = postProcess(results, imgHeight, imgWidth)
    print('Segmentation done!')
    
    # Generate coordinate points of the contour of the segmented image
    response_dic = generatePoints(segImage)
    print('Points of the contour generated!')
    print(response_dic)
    
    # Create a response with the JSON representation and send back to client side
    response = jsonify(response_dic)
    response.headers['Access-Control-Allow-Origin']='*'
    print('X and Y coordinate points are sent back to client side!')
    return response


''' 
This is Store Annotation REST API that requests image data from client side,
generates the contours of the annotation and stores the annotated image with original image name,
size and all coordinates of points with lesion type in csv file in the local storage (flask-app folder) 
'''

@app.route('/store-annotation', methods = ['POST'])
def store_annotation():
    # Request and get json data
    req_data = request.get_json()
    print('Data received!')

    # Get the image name, size of the image and lesion type from json data
    imgName = (req_data.get("imgName", "none"))
    imgSize = (req_data.get("imgSize", "none"))
    lesionType = (req_data.get("lesionType", "none"))
	#imgView = (req_data.get("imgView", "none"))
    print('Image name: ', imgName)
    print('Image size: ', imgSize)
    print('Type of lesion: ', lesionType)
	#print('Image view: ', imgView)
    
    # Get and generate x and y coordinates of points of annotated contours from json data
    main_contour, coord_x = generateContourPoints(req_data)
    print('Main numpy contour: ', main_contour)
    
    # Make and fill contour with white pixels using points of the contours
    img = np.zeros(imgSize) # create a single channel black image
    for i in range(len(coord_x)):
        cv2.fillPoly(img, pts=[np.array(main_contour[i])], color=255)
    img = img.astype(int)
    
    # Save annotated image in the local storage according to lesion type
    annotation_folder = './store_annotation/'
    benign_folder = 'benign/'
    malignant_folder = 'malignant/'
    if lesionType == 'les':
        cv2.imwrite(annotation_folder + imgName + '_' + lesionType + '.png',img)
    elif lesionType == 'ben':
        cv2.imwrite(annotation_folder + benign_folder + imgName + '_' + lesionType + '.png',img)
    else:
        cv2.imwrite(annotation_folder + malignant_folder + imgName + '_' + lesionType + '.png',img)
    print('Annotation saved!')
    
    # Save all coordinates of points with image name, size and lesion type in CSV file later to use
    w = csv.writer(open(annotation_folder + imgName + '.csv', "w"))
    for key, val in req_data.items():
        w.writerow([key, val])
    print('Annotation informations saved into CSV file!')
    
    # Save all annotation informations into MongoDB database
    client = MongoClient('localhost', 3001)
    db = client['store_annotation_db']
    collection = db['annotation_info']
    collection.insert_one(req_data)
    client.close()
    print('Annotation informations saved in json format into MongoDB database!')
    
    # Create a response with the JSON representation and send back to client side
    response_dic = {'msg' : 'Annotation successfully stored!'}
    response = jsonify(response_dic)
    response.headers['Access-Control-Allow-Origin']='*'
    return response


if __name__ == '__main__':
    app.run(debug=True)