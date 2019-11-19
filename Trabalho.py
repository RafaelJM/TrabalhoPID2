"""
@author: Rafael Junqueira Martarelli
@language: Python3.7
"""

import cv2 as cv
import glob
    
#--- images
def cropImages(images):
    for i, image in enumerate(images):
        image = cv.imread(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            cv.rectangle(image, (x, y), (x + w, y + h), 2)
        
        images[i] = image
    
    return images

# --- Extração de caracteristicas
def LBP():
    return 0

def PCA():
    return 0

def LDA():
    return 0

# --- Metodos
def EER():
    return 0

def ROC():
    return 0

def CMC():
    return 0

def CurvasPrecisaoRevocacao():
    return 0

def FMeasure():
    return 0

# --- Testes
def test1():
    files = glob.glob("dataset/*.bmp")
    numberClass = []
    for file in files:
        
    return files
        

# 2 caracteriscas
# todos metodos
# tastar com pelo menos 2 de luz e obstrução de face
# usar um tal de cmc para ver o q o dataset traz como mais proximo (cmc database recuperation)