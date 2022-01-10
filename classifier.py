import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import PIL.ImageOps
import os,ssl

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image


X, y = fetch_openml('mnist_784',version = 1,return_X_y = True)

xtrain , xtest ,ytrain , ytest = train_test_split(X,y,random_state = 9 , train_size = 7500 , test_size = 2500)
xtrainScaled = xtrain / 255.0
xtestScaled = xtest / 255.0
clf = LogisticRegression(solver = 'saga',multi_class = 'multinomial').fit(xtrainScaled , ytrain)

def get_prediction(image):
    impil = Image.open(image)
    imgbw = impil.convert('L')
    imgbwresize = imgbw.resize((28,28),Image.ANTIALIAS)

    pixelfilter = 20
    minpixel = np.percentile(imgbwresize , pixelfilter)
    imginvertedscale = np.clip(imgbwresize - minpixel ,0 ,255)
    maxpixel = np.max(imgbwresize)
    imginvertedscale = np.asarray(imginvertedscale)/maxpixel
    testsample = np.array(imginvertedscale).reshape(1,784)
    testpred = clf.predict(testsample)
    return testpred[0]

    


