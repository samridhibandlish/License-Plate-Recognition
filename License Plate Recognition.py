import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skimage.feature import hog
import imutils
import keras
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

train_x = []
train_y = []
test_x = []
test_y = []

labels = ['license', 'no_license']

# Loads training and test data by their labels
def load_images(dir_path_im):
    images = []
    for i in labels:
        for j in os.listdir(os.path.join(dir_path_im, i)):
            image = cv2.imread(os.path.join(os.path.join(dir_path_im, i), j))
            images.append([image, i])
    return images

# path of train and test data
train_data = load_images('..\\data\\training')
test_data = load_images('..\\data\\testing')

# separating the image and label in different sets
# switches the label to a numerical value
for image, label in train_data:
    train_x.append(image)
    
    # 0 = no license, 1 = license
    new_label = 0
    if (label == 'license'):
        new_label = 1
    train_y.append(new_label)
    
# converting the label list to an array
train_y = np.asarray(train_y)

for image, label in test_data:
    test_x.append(image)
    new_label = 0
    if (label == 'license'):
        new_label = 1
    test_y.append(new_label)
test_y = np.asarray(test_y)

new_trainx = []
blur_trainx = []
resize_trainx = []
new_label = []
new_testx = []

# preprocessing the image 
# and appending it to different arrays
for img in range(0, len(train_x)):
    
    # resizing the image to be uniform
    resize_img = cv2.resize(train_x[img], (300, 200))
    resize_trainx.append(resize_img)
    
    # converting image to grayscale
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

    # smoothening the image
    blur_img = cv2.bilateralFilter(gray_img, 13, 15, 15)
    blur_trainx.append(blur_img)

    # finding edges in the image
    image_filter = cv2.Canny(blur_img, 30, 200)
    new_trainx.append(image_filter)

# resizing the images in the test set
for img in range(0, len(test_x)):
    resize_img = cv2.resize(test_x[img], (300, 200))
    new_testx.append(resize_img)

plate_trainx = []
license_trainx = []
rectx = []

# feature extraction on the image
for img in range(0, len(new_trainx)):
    
    # find the contours in the image
    contours = cv2.findContours(new_trainx[img], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    count = None
    new_lbl = 0

    for contor in contours:

        # approximate the largest contour (rectangular)
        para = cv2.arcLength(contor, True)
        approx = cv2.approxPolyDP(contor, 0.01 * para, True)
        if len(approx) == 4:
            count = approx
            break

    # if a rectangle is detected
    if count is not None:
        
        contour_rect = resize_trainx[img]
        contour_rect = cv2.drawContours(contour_rect, [count], contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        rectx.append(contour_rect)
        
        # mask the image and draw the rectangle on it
        mask = np.zeros(blur_trainx[img].shape, np.uint8)
        mask = cv2.drawContours(mask, [count], 0, 255, -1)

        # extract the license plate
        plate = cv2.bitwise_and(resize_trainx[img], resize_trainx[img], mask=mask)
        plate_trainx.append(plate)

        # change the label to 1
        new_lbl = 1

        # reads the license plate of an image
        license_number = pytesseract.image_to_string(plate)
        if license_number != '':
            license_trainx.append(license_number)
        else:
            license_trainx.append("0")

    # if no rectangle detected
    else:
        plate_trainx.append(resize_trainx[img])
        
        # change the label to 0
        new_lbl = 0

    new_label.append(new_lbl)

# SVM on HOG
HOG_train = []
for i in range(0,len(plate_trainx)):
    image = plate_trainx[i]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(image, orientations = 8, pixels_per_cell = (5,5), cells_per_block = (2,2), visualize = True)
    HOG_train.append(fd)

HOG_test = []
for i in range(0,len(new_testx)):
    image = new_testx[i]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(image, orientations = 8, pixels_per_cell = (5,5), cells_per_block = (2,2), visualize = True)
    HOG_test.append(fd)

# creating a model
classifier = SVC(kernel='linear')
classifier.fit(HOG_train, train_y)

# testing the model
y_pred = classifier.predict(HOG_test)

# printing the classification report
print(classification_report(test_y, y_pred))