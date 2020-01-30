import numpy as np
import cv2


''' These function converts 16 bit image to 8 bit and rescales between 0-255 '''

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



''' This function generates coordinate points of the contour of the segmented image
and returns in the dictionary format. If image does not have a lesion, 
returns message "No segmentation!" '''

def generatePoints(segImage):
    if cv2.countNonZero(segImage) == 0:
        response_dic = {'msg' : 'Image does not have a lesion! No segmentation!'}
        return response_dic
    else:
        im2, contours, hierarchy = cv2.findContours(segImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #drawnContour_img = cv2.drawContours(segImage, contours, -1, (0,255,0), 3)
        #contoured = cv2.drawContours(segImage, contours, -1, (255,0,0), 3)
        print('Contour of segmentation is found!')
        #print('Actual number of points of contour: ' + str(len(contours[0])))
        
        xContourCoordPoints = []
        yContourCoordPoints = []
            
        for j in range(len(contours)):
            xCoordPoints = []
            yCoordPoints = []
            # reduce coordinate points of the contour and keep at least 3 points
            if (len(contours[j]) > 250):
                coordinate_reduce_step = 4
            elif (len(contours[j]) > 24):
                coordinate_reduce_step = 3
            elif (len(contours[j]) > 9):
                coordinate_reduce_step = 2
            else:
                coordinate_reduce_step = 1
            for i in range(0, len(contours[j]), coordinate_reduce_step):
                xCoordPoints.append(contours[j][i][0][0])
                yCoordPoints.append(contours[j][i][0][1])
            xContourCoordPoints.append(np.copy(xCoordPoints[0:len(xCoordPoints)]))
            yContourCoordPoints.append(np.copy(yCoordPoints[0:len(yCoordPoints)]))
        
        response_dic = {'msg' : 'Segmentation successfully completed!'}
        for i in range(len(xContourCoordPoints)):
            response_dic.update( {'xCoordContour' + str(i) : xContourCoordPoints[i].tolist()} )
            response_dic.update( {'yCoordContour' + str(i) : yContourCoordPoints[i].tolist()} )
        return response_dic



''' This function gets first x and y coordinates of points of annotated contour from json data,
then generates and return in the list '''

def generateContourPoints(req_data):
    coord_x = []
    coord_y = []
    for i in range(1, (int((len(req_data)-2)/2)+1)):
        coord_x.append(req_data.get(('x'+str(i)), "none"))
        coord_y.append(req_data.get(('y'+str(i)), "none"))
    print(coord_x)
    print(coord_y)
    main_contour = []
    for i in range(len(coord_x)):
        contours = []
        # if annotated contour is only one
        if len(coord_x)==1:
            for j in range(len(coord_x[i])):
                contours.append([coord_x[i][j], coord_y[i][j]])
                if len(coord_x[i])-1 == j:
                    main_contour.append(contours)
        else:
            for j in range(len(coord_x[i])):
                contours.append([coord_x[i][j], coord_y[i][j]])
                if len(coord_x[i])-1 == j:
                    main_contour.append(contours)
    return main_contour, coord_x
