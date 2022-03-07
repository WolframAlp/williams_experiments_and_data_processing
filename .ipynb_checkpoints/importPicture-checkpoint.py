# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 09:06:26 2021

@author: Gusta
"""
import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
#import time
from mpl_toolkits.mplot3d import Axes3D




# Imports all .tif images from a folder.
# Input: path, the path to the folder where the images a saved. Will download images if neacesary.
# Output: image3DMatrix, 3D matrix with all the image saved in the format (image,y,x)
def importImage3DMatrix(path):
    os.chdir(path)
    fileType = '*.tif'
    fullPath = os.path.join(path, fileType)
    imageList = []
   
    # Find all .tif files a put them into a list of images.
    for filename in glob.glob(fullPath): #assuming tif
        tempImage = Image.open(filename)
        imageList.append(tempImage)
      
    image3DMatrix = np.empty([len(imageList), 1456, 1936], )
     
    for i in range(len(imageList)):
        image3DMatrix[i] = np.asarray(imageList[i], dtype = 'uint8')

    image3DMatrix = image3DMatrix.astype(np.uint8)

    return image3DMatrix
            


# Find the minimum and maximum valuesn values of the pixel in an image
# Input: image2DMatrix, the image as a np.matrix
# Output: [minValue, maxValue], vector with the minimum value and maximum value of image
def minMaxValues(image2DMatrix):
    [minValue, maxValue] = int(np.min(image2DMatrix)), int(np.max(image2DMatrix))
    return [minValue, maxValue]

# Linear grascale mapping used for histogram streching
# Input: minD, the minimum desired value in histogram (usually 0)
# Input: maxD, the maximum desired value in histogram (usually 255)
# Input: minImage, the minimum valued pixel in the image
# input: maxImage, the maximum valued pixed in the image
# Input: imageMateix, the image as a np.matrix
# Output: newImageMatrix, new np.matrix of image with streched values
def linearGrayLevelMapping(minD, maxD, minImage, maxImage, imageMatrix):
    newImageMatrix = np.multiply(((maxD - minD) / (maxImage - minImage)),(imageMatrix - minImage)) + minD
    newImageMatrix = newImageMatrix.astype(np.uint8)
    return newImageMatrix


# Linear grascale mapping used for histogram streching, with automatic min/max value finding
# Input: minD, the minimum desired value in histogram (usually 0)
# Input: maxD, the maximum desired value in histogram (usually 255)
# Input: imageMateix, the image as a np.matrix
# Output: newImageMatrix, new np.matrix of image with streched values
def linearGrayLevelMappingAutomatic(minD, maxD , imageMatrix):
    [minImage, maxImage] = minMaxValues(imageMatrix)
    newImageMatrix = np.multiply(((maxD - minD) / (maxImage - minImage)),(imageMatrix - minImage)) + minD
    newImageMatrix = newImageMatrix.astype(np.uint8)
    return newImageMatrix

# Makes it so there is zero padding around the image
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: layers, number of padding in rows and columns 
# Output: paddedImage, 2D matrix of the original data with zero padding
def zeroPadding(image2DMatrix, layers):
    paddedImage = np.zeros((len(image2DMatrix) + 2 * layers, len(image2DMatrix[0]) + 2 * layers))
    for i in range(len(image2DMatrix)):
        for j in range(len(image2DMatrix[0])):
            paddedImage[i + layers, j + layers] = image2DMatrix[i,j]
    return paddedImage
    

# Makes it so there is same padding around the image
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: layers, number of padding in rows and columns 
# Output: paddedImage, 2D matrix of the original data with same padding
def samePadding(image2DMatrix, layers):
    paddedImage = zeroPadding(image2DMatrix, layers)
    lengthMatrixY = len(paddedImage)
    lengthMatrixX = len(paddedImage[0])
    
    for i in range(layers,0,-1):
        paddedImage[i - 1,:] = paddedImage[i,:]
        paddedImage[:,i - 1] = paddedImage[:,i]
        paddedImage[lengthMatrixY - i ,:] = paddedImage[lengthMatrixY - i - 1,:]
        paddedImage[:, lengthMatrixX - i] = paddedImage[:, lengthMatrixX - i - 1]
    return paddedImage
        

# Takes the 3D matrix of all images and finds the average value in every pixel
# input: image3DMatrix, 3D matrix of several images in the format (z,y,x) (defult format)
# Output: meanMatrix, 2D matrix with the mean value of every image
def avgMatrix(image3DMatrix):
    meanMatrix = np.divide(np.sum(image3DMatrix, 0), len(image3DMatrix))
    meanMatrix = meanMatrix.astype(np.uint8)
    return meanMatrix


# Median filter used to filter out salt and peber noice
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: filterSize, size of the filter, e.g. filterSize = 3 mean 3x3 filter, must be an odd number
# Input: lowerThreshold, the lower threshold values which the filter reacts to
# Input: upperThreshold, the upper threshold values which the filter reacts to
# Input: paddingChoice, choice of padding for the image, defult option is same padding
#        if paddingChoice = 0, then zero padding will be used
#        if paddingChoice = 1, then same padding will be used
def MedianFilterSingleImage(image2DMatrix, filterSize, lowerThreshold, upperThreshold, paddingChoice = 1):
    if (filterSize % 2 != 1):
        return "FilterSize must be odd"
    paddingSize = int((filterSize - 1) / 2)
    
    if paddingChoice == 0:
        paddedImage = zeroPadding(image2DMatrix, paddingSize)
    elif paddingChoice == 1:
        paddedImage = samePadding(image2DMatrix, paddingSize)
    else:
         return "Please choose one of the two styles of padding"   
    
    filteredImage = np.copy(paddedImage)
    
    lengthMatrixY = len(paddedImage)
    lengthMatrixX = len(paddedImage[1])
    
    for i in range(paddingSize, lengthMatrixY - paddingSize):
        for j in range(paddingSize, lengthMatrixX - paddingSize):
                if paddedImage[i,j] <= lowerThreshold or paddedImage[i,j] >= upperThreshold:
                    tempMatrix = paddedImage[i - 1 : i + 2, j - 1 : j + 2]
                    filteredImage[i,j] = np.median(tempMatrix)
#    
    filteredImage = filteredImage[paddingSize : lengthMatrixY - paddingSize, paddingSize : lengthMatrixX - paddingSize]
    
    return filteredImage
                    



# Function that takes an image a print the histogram of that image
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: threshold, threshold for the values showed in histogram, defult = 0
# Input: histMax, the highest value the histogram will show, defult = 255
# Output: Plottet histogram of image
# Warning: If there is no values above threshold the function will crash
def plotHistogram(image2DMatrix, threshold = 0, histMax = 255):
    # Since the function np.histogram takes the values in the half open interval between two numbers in the range
    # the function will always have one more bin than intervals. Therefore the range needs to go to 257. This is 
    # also the reason the last index is removed in the plot since there are no values in the interval [256, infinity)
    [histY, histX] = np.histogram(image2DMatrix, bins = range(257))
    findZero = np.where(histY == 0)[0]
    histXZeroRemoved = np.delete(histX, findZero)
    histYZeroRemoved = np.delete(histY, findZero)
    
    findThreshold = np.where(histX < threshold)[0]
    histXFinal = np.delete(histXZeroRemoved, findThreshold)
    histYFinal = np.delete(histYZeroRemoved, findThreshold)
    
    # Removing the extra value from the X part of the histogram
    histXFinal = histXFinal[:-1]
    
    plt.figure('Histogram of image')
    plt.plot(histXFinal, histYFinal)
    # The plot might stop before 255, however it is chosen since the possible pixelvalues can go to 255 which makes 
    # histograms easier to compare.If the plot stops before 255 it mean that there are no pixels with that value.
    plt.xlim([threshold, histMax])
    plt.ylim([0, np.max(histYFinal[histXFinal >= threshold])])
    plt.show()
    return
    

# Makes a mean histogram of all the images.
# input: image3DMatrix, 3D matrix of several images in the format (z,y,x) (defult format)
# Input: threshold, threshold for the values showed in histogram, defult = 0
# Input: histMax, the highest value the histogram will show, defult = 255
# Output: Plottet histogram of the average image
# Warning: If there is no values above threshold the function will crash
def mean3DHistogram(image3DMatrix, threshold = 0, histMax = 255):
    meanMatrix = avgMatrix(image3DMatrix)
    meanMatrix = meanMatrix.astype(np.uint8)
    plotHistogram(meanMatrix, threshold, histMax)
    return

# Finds the coordinates of the upper left corner and the bottom right corner in a smallest rectangle that covers all threshold values.
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: lowerThreshold, smallest value that the rectangle must cover
# Output: coordinates, vector with coordinates for upper left corner og lower right corner of rectangle
#         upper left corner = coordinates[0], coordinates[1]
#         lower left corner = coordinates[2], coordinates[3]
def findSpotRectangle(image2DMatrix, lowerThreshold):
    lengthY = len(image2DMatrix)
    lengthX = len(image2DMatrix[0])
    
    tempIndexX = np.empty(0)
    for i in range(lengthY):
        indexThresholdValueX = np.flatnonzero(image2DMatrix[i,:] >= lowerThreshold)
        tempIndexX = np.append(tempIndexX, indexThresholdValueX)
    indexX = [np.min(tempIndexX), np.max(tempIndexX)]
    
    tempIndexY = np.empty(0)
    for i in range(lengthX):
        indexThresholdValueY = np.flatnonzero(image2DMatrix[:,i] >= lowerThreshold)
        tempIndexY = np.append(tempIndexY, indexThresholdValueY)
    indexY = [np.min(tempIndexY), np.max(tempIndexY)]
    
    coordinates = [indexX[0], indexY[0], indexX[1], indexY[1]]
    return coordinates
    

# Draws a white rectangle around around the spot with the given threshold
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: lowerThreshold, smallest value that the rectangle must cover
# Output drawing, 2D Matrix of image with white rectangle around threshold values
def drawSpotRectangle(image2DMatrix, lowerThreshold):
    drawing = np.copy(image2DMatrix)
    coordinates = findSpotRectangle(drawing, lowerThreshold)
    for i in range(int(coordinates[0]), int(coordinates[2]) + 1):
        drawing[int(coordinates[1]), i] = 255
        drawing[int(coordinates[3]), i] = 255
    for i in range(int(coordinates[1]), int(coordinates[3]) + 1):
        drawing[i, int(coordinates[0])] = 255
        drawing[i, int(coordinates[2])] = 255
    return drawing



# Finds the area of the rectangle around the spot with the given threshold
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: lowerThreshold, smallest value that the rectangle must cover
# Output: area, area of the rectangle
# Output: areaComparedToImage, area of rectangle compared to the area of the image
# Output: trueValueRectangle, percentage of pixels in rectangle which is greater than threshold value
def areaSpotRectangle(image2DMatrix, lowerThreshold):
    coordinates = findSpotRectangle(image2DMatrix, lowerThreshold)
    
    firstX = int(coordinates[0])
    lastX = int(coordinates[2])
    firstY = int(coordinates[1])
    lastY = int(coordinates[3])
    
    diffX = lastX - firstX
    diffY = lastY - firstY
    
    area = (diffX + 1) * (diffY + 1)
    areaComparedToImage = area / ((len(image2DMatrix[0]) + 1) * (len(image2DMatrix[1]) + 1))
    
    matrixRectangle = image2DMatrix[firstY:lastY + 1, firstX:lastX + 1]
    trueValueCounter = np.sum(matrixRectangle >= lowerThreshold)
    trueValueRectangle = trueValueCounter / area
    
    return [area, areaComparedToImage, trueValueRectangle]

    
    
    
# Find the spot which will be defined as the center, this is done by looking at the sum in a 5x5 area
# the area with the highest sum is defined as the center, if more that one point with the maximum sum
# then the point with the highest original value will be used. IF this is the same, a random one of the 
# point will be chosen as center point.
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: paddingChoice, choice of padding for the image, defult option is same padding
#        if paddingChoice = 0, then zero padding will be used
#        if paddingChoice = 1, then same padding will be used
# Output: indexMax, gives the index of the 
# Warning: if more than one point has a center value the function will output the first one
def findCenterSpot(image2DMatrix, paddingChoice = 1):
    
    if paddingChoice == 0:
        paddedImage = zeroPadding(image2DMatrix, 2)
    elif paddingChoice == 1:
        paddedImage = samePadding(image2DMatrix, 2)
    else:
         return "Please choose one of the two styles of padding" 
     
    lengthX = len(paddedImage[0])
    lengthY = len(paddedImage)
    sumMatrix = np.zeros([lengthY,lengthX])
    
    for i in range(2, lengthY - 2):
        for j in range(2, lengthX - 2):
            tempMatrix = paddedImage[j - 2 : j + 3, i - 2 : i + 3]
            sumMatrix[i,j] = np.sum(tempMatrix)
    
    sumMatrix = sumMatrix[2 : lengthY - 2, 2 : lengthX - 2]
    maxValueCounter = np.count_nonzero(sumMatrix == np.max(sumMatrix))
    
    if maxValueCounter > 1:
        print("More than one max value, number of max values is: ", maxValueCounter)
    
    indexMax = np.unravel_index(np.argmax(sumMatrix, axis=None), sumMatrix.shape)
    indexMax = np.array(indexMax)
    
    return indexMax
    



# Makes a black/white image of the original image.
# Input: imageMatrix, image as an np.matrix, all dimensions allowed
# Input: threshold, gives threshold value such that everything under the thresholdvalue
#        becomes black and everything equal or above becomes white
# Output: blobImage, image of blob according to chosen threshold
def BLOB(imageMatrix, threshold):
    blobImage = np.copy(imageMatrix)
    blobImage[blobImage >= threshold] = 255
    blobImage[blobImage < threshold] = 0
    
    return blobImage


# Finds the center spot of every image in a dataset. The area the image looks for the center is significantly smaller
# than the findCenterSpot function, this is done to save time on bigger datasets.
# Input: image3DMatrix, 3D matrix of several images in the format (z,y,x) (defult format)
# Output: centerSpot, n x 2 matrix with the coordiantes for the center spots of every image, in the format (y,x)
def centerOfDataSet(image3DMatrix):
    centerSpot = np.zeros([len(image3DMatrix), 2])
    
    for i in range(len(image3DMatrix)):
        
        coordinates = findSpotRectangle(image3DMatrix[i], np.max(image3DMatrix[i]) - int((1/4) * np.max(image3DMatrix[i]))) 

        firstX = int(coordinates[0])
        lastX = int(coordinates[2])
        firstY = int(coordinates[1])
        lastY = int(coordinates[3])
        
        tempData = image3DMatrix[i, firstY:lastY + 1, firstX:lastX + 1]       
        tempCenterSpot = findCenterSpot(tempData)    
        centerSpot[i, 0] = tempCenterSpot[0] + firstX
        centerSpot[i, 1] = tempCenterSpot[1] + firstY
        
        # Used to check progress since this function can take a long time.
#        print("Done with image number; ", i)
        
    return centerSpot

# Finds the average pixel value for every image in the 3D matrix
# Input: image3DMatrix, 3D matrix of several images in the format (z,y,x) (defult format)
# Output: avgPixel, array with average pixel values. 
def avgPixelValue(image3DMatrix):
    zyMatrix = np.sum(image3DMatrix,2)
    zMatrix = np.sum(zyMatrix,1)
    avgPixel = np.divide(zMatrix, len(image3DMatrix[0,0,:]) * len(image3DMatrix[0,:,0]))
    
    return avgPixel
    
# Code is heavely based on code from: https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
# Accesed on: 17/02/2021 
# Makes an elliptic fit to the data
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: threshold, value that the ellipse is based on
# Output: formulaEllispse, the formuale of the ellipse equation on the form:
#         formulaEllispse[0]*x^2 + formulaEllispse[1] *x*y + formulaEllispse[2] * y^2
#         + formulaEllispse[3] * x + formulaEllispse[4] * y
def ellipseFunction(image2DMatrix, threshold):
    coordinates = findSpotRectangle(image2DMatrix, threshold)
    firstX = int(coordinates[0])
    lastX = int(coordinates[2])
    firstY = int(coordinates[1])
    lastY = int(coordinates[3])
    
    tempData = image2DMatrix[firstY:lastY + 1, firstX:lastX + 1]
    
    tempIndex = np.where(tempData == threshold)
    
    X = tempIndex[1] + firstX
    Y = tempIndex[0] + firstY
    
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)
    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    formulaEllispse = np.linalg.lstsq(A, b, rcond = 0)[0].squeeze()
    
    return formulaEllispse

# Code is heavely based on code from: https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
# Accesed on: 17/02/2021 
# Makes an elliptic fit to the data and plots it
# Input: image2DMatrix, 2D Matrix of a singular image
# Input: threshold, value that the ellipse is based on
# Output: Scatter plot, ellipse plottet with threshold values.
def ellipseFitWithPlot(image2DMatrix, threshold):
    coordinates = findSpotRectangle(image2DMatrix, threshold)
    firstX = int(coordinates[0])
    lastX = int(coordinates[2])
    firstY = int(coordinates[1])
    lastY = int(coordinates[3])
    
    tempData = image2DMatrix[firstY:lastY + 1, firstX:lastX + 1]
    
    tempIndex = np.where(tempData == threshold)
    
    X = tempIndex[1] + firstX
    Y = tempIndex[0] + firstY
    
    X = X.reshape(len(X), 1)
    Y = Y.reshape(len(Y), 1)
    # Formulate and solve the least squares problem ||Ax - b ||^2
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    formulaEllispse = np.linalg.lstsq(A, b, rcond = 0)[0].squeeze()

    # Print the equation of the ellipse in standard form
#    print('The ellipse is given by {0:.3}x^2 + {1:.3}xy+{2:.3}y^2+{3:.3}x+{4:.3}y = 1'
#          .format(formulaEllispse[0], formulaEllispse[1],formulaEllispse[2],formulaEllispse[3],formulaEllispse[4]))

    # Plot the noisy data
    s = [1 for n in range(len(X))]
    plt.figure()
    plt.imshow(image2DMatrix, cmap='gray')
    plt.scatter(X, Y, s, label='Data Points')
   
    # Plot the least squares ellipse
    x_coord = np.linspace(firstX - int(0.01 * firstX), lastX + int(0.01 * lastX), 1000)
    y_coord = np.linspace(firstY - int(0.01 * firstY), lastY + int(0.01 * lastY), 1000)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = (formulaEllispse[0] * X_coord ** 2 + formulaEllispse[1] * X_coord * Y_coord + formulaEllispse[2] 
    * Y_coord**2 + formulaEllispse[3] * X_coord + formulaEllispse[4] * Y_coord)
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
    
    # Axis are made equal to not visually confuse user
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    
    return


def make_hist(im, axis=0, plot=False):
    s = np.sum(im,axis=axis) / im.shape[axis]
    
    if plot:
        plt.figure()
        plt.plot(np.arange(im.shape[1 - axis]), s)
    
    return s


def get_image(file):
    if file[-4:] == ".bmp":
        Image.open(file).save(file[:-4] + ".png")
        file = file[:-4] + ".png"
    return  np.array(Image.open(file))[:,:,0]
    

if __name__ == '__main__':
    
    di = "D:\\OneDrive - Danmarks Tekniske Universitet\\OneDrive\\Dokumenter\\DTU\\Terahertz\\PaperExperiments\\measurements\\Spectrometer_Chip\\070921\\1000GHz_current\\"
    fi = "Image-000003.bmp"
    im = get_image(di + fi)
    xsum = make_hist(im)
    ysum = make_hist(im, axis=1)

    xmax, ymax = xsum.max(), ysum.max()
    
    val = int(round((xmax + ymax) / 6, 0))
    ellipseFitWithPlot(im,val)

# This code is originally from:
# https://stackoverflow.com/questions/52385299/plot-a-3d-bar-histogram-with-python
    
#os.chdir('D:/Gustav_data')
#
#a_cen = np.loadtxt('99_to_100_percent_power_100_images_center.txt', delimiter = ',')
#
## To generate some test data
#x = a_cen[:,0]
#y = a_cen[:,1]
#
#XY = np.stack((x,y),axis=-1)
#
#def selection(XY, limitXY=[[-2,+2],[-2,+2]]):
#        XY_select = []
#        for elt in XY:
#            if elt[0] > limitXY[0][0] and elt[0] < limitXY[0][1] and elt[1] > limitXY[1][0] and elt[1] < limitXY[1][1]:
#                XY_select.append(elt)
#
#        return np.array(XY_select)
#
#XY_select = selection(XY, limitXY=[[400,1200],[400,1200]])
#
#heatmap, xedges, yedges = np.histogram2d(XY_select[:,0], XY_select[:,1], bins = 401, range = [[400,1200],[400,1200]])
#extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#
#
#plt.figure("Histogram")
##plt.clf()
#plt.imshow(np.log(heatmap.T +1), extent=extent, origin='lower', cmap = 'gray_r')
#plt.colorbar()
#plt.show()
#
#
#plt.figure("Histogram 1")
##plt.clf()
#plt.imshow(heatmap.T, extent=extent, origin='lower', cmap = 'gray_r')
#plt.colorbar()
#plt.show()
#
#


