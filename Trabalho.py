"""
@author: Rafael Junqueira Martarelli
@language: Python3.7
"""

import cv2 as cv
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

#--- tools
def saveDataBase():
    dataset = fetch_lfw_people(min_faces_per_person=100, resize=0.5, color = False)
    for i, img in enumerate(dataset.images):
        cv.imwrite("database/"+str(i)+".jpg",img)

def PCA(Data_train, Data_test):
    pca = sklearn_PCA().fit(Data_train)
    return (pca.transform(Data_train), pca.transform(Data_test))

# --- Extração de caracteristicas
def LBP():
    return 0

def PCA_SVC(Data_train, Data_test, Target_train, Target_test):
    Data_train, Data_test = PCA(Data_train, Data_test)
    svc = SVC()
    svc = svc.fit(Data_train, Target_train)
    return confusion_matrix(Target_test, svc.predict(Data_test))

def PCA_KNeighbors(Data_train, Data_test, Target_train, Target_test):
    Data_train, Data_test = PCA(Data_train, Data_test)
    kNeighbors = KNeighborsClassifier()
    kNeighbors.fit(Data_train, Target_train)
    return confusion_matrix(Target_test, kNeighbors.predict(Data_test))

def PCA_ML(Data_train, Data_test, Target_train, Target_test):
    Data_train, Data_test = PCA(Data_train, Data_test)
    ml = MLPClassifier()
    ml.fit(Data_train, Target_train)
    return confusion_matrix(Target_test, ml.predict(Data_test))

def LDA(Data_train, Data_test, Target_train, Target_test):
    lda = LinearDiscriminantAnalysis().fit(Data_train, Target_train)
    return confusion_matrix(Target_test, lda.predict(Data_test))

# --- Metodos
def EER(): #https://stackoverflow.com/questions/28339746/equal-error-rate-in-python #https://pythonhosted.org/bob/temp/bob.measure/doc/py_api.html
    return 0

def ROC(): #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html #https://pythonhosted.org/bob/temp/bob.measure/doc/py_api.html
    
    return 0

def CMC(): #https://pythonhosted.org/bob/temp/bob.measure/doc/py_api.html
    return 0

def CurvasPrecisaoRevocacao():
    return 0

def FMeasure(): #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    return 0

# --- Testes
def test1():
    dataset = fetch_lfw_people(min_faces_per_person=100, resize=0.5, color = False)
    Data_train, Data_test, Target_train, Target_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state= None)
    LDA_CM = LDA(Data_train, Data_test, Target_train, Target_test)
    ML_CM = PCA_ML(Data_train, Data_test, Target_train, Target_test)
    return (dataset, LDA_CM, ML_CM)

teste_1 = test1()

# 2 caracteriscas
# todos metodos
# tastar com pelo menos 2 de luz e obstrução de face
# usar um tal de cmc para ver o q o dataset traz como mais proximo (cmc database recuperation)