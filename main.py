# File: main.py
# Description: Basic Document Scanenr written in Python using OpenCV
# Date: January 2023
# Author: Steven Wong

import cv2 as cv
import numpy as np

# Use a series of erosion and dilation operations remove text for edge detection
def removeText(image):
    kernel = np.ones((5,5),np.uint8)
    # Repeatedly apply erosion and dilation to remove text
    image_dilation = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=10)
    return image_dilation

# Use GrabCut to remove the background of the image
def grabCut(image):
    # Take the corners of the image as the background
    mask = np.zeros(image.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,image.shape[1]-20,image.shape[0]-20)
    cv.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    image = image*mask2[:,:,np.newaxis]
    return image

# Read our receipt file as an image
img = cv.imread('receipt.png')
cv.imshow('Original', img)
cv.waitKey(0)

# Make a copy of the image for pre-processing
img_copy = img.copy()
# Remove the text from the image
img_copy = removeText(img_copy)
img_copy = grabCut(img_copy)
cv.imshow('Pre-Processing', img_copy)
cv.waitKey(0)

# Apply Gaussian Blur to the image to reduce noise in the image
img_copy = cv.GaussianBlur(img_copy, (11, 11), 0)
# Use the Canny Edge Detection algorithm to detect the edges of the document
img_copy = cv.cvtColor(img_copy, cv.COLOR_BGR2GRAY)
img_copy = cv.Canny(img_copy, 0, 200)
# Dilate the image to make the edges thinner
img_copy = cv.dilate(img_copy, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
cv.imshow('Canny Edge Detection', img_copy)
cv.waitKey(0)

# Perform contour detection to get the contours of these edges and sort them by area


# OCR the text: Once you have transformed the image, you can use Optical Character Recognition (OCR) to extract the text from the document. 
# There are several OCR libraries available for Python, such as Tesseract and pytesseract.

# Post-process the text: Finally, you may need to post-process the text to correct any errors or inconsistencies introduced by the OCR process.
#  This may include tasks such as spelling correction and layout analysis.