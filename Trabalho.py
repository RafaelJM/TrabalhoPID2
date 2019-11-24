"""
@author: Rafael Junqueira Martarelli
@language: Python3.7
"""

import cv2 as cv
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as sklearn_PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import label_binarize
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC

#--- tools
def writeDataBase(Data,Target,imageShape,min_faces_per_person):
    for i, img in enumerate(Data):
        cv.imwrite("database/"+str(Target[i])+"-"+str(i%min_faces_per_person)+".jpg",img.reshape(imageShape))

def getDatas(min_faces_per_person, test_size, saveDataBase = False):
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color = False)
    Data = dataset.data[dataset.target == 0][:min_faces_per_person]
    Target = dataset.target[dataset.target == 0][:min_faces_per_person]
    for i in range(1,len(dataset.target_names)):
        Data = np.concatenate((Data,dataset.data[dataset.target == i][:min_faces_per_person]))
        Target = np.concatenate((Target,dataset.target[dataset.target == i][:min_faces_per_person]))
    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, label_binarize(Target, classes = list(range(len(dataset.target_names)))), test_size=test_size, random_state= None)
    imageShape = (len(dataset.images[0]), len(dataset.images[0][0]))
    if saveDataBase:
        writeDataBase(Data,Target,imageShape,min_faces_per_person)
    return (Data_train, Data_test, Target_train, Target_test, imageShape)

# --- Extratores de caracteristicas e Classificadores

def PCA(Data_train, Data_test):
    pca = sklearn_PCA().fit(Data_train)
    return (pca.transform(Data_train), pca.transform(Data_test))
 
def LBP(Data_train, Data_test, imageShape, numPoints = 27, radius = 3):
    Data_TR = []
    Data_TE = []
    for i in range(len(Data_train)):
        lbp = local_binary_pattern(Data_train[i].reshape(imageShape), numPoints, radius, 'uniform')
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        Data_TR.append(hist)
    for i in range(len(Data_test)):
        lbp = local_binary_pattern(Data_test[i].reshape(imageShape), numPoints, radius, 'uniform')
        (hist, _) = np.histogram(lbp.ravel(),bins=np.arange(0, numPoints + 3),range=(0, numPoints + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        Data_TE.append(hist)
    return (Data_TR, Data_TE)

def PCA_ML(Data_train, Data_test, Target_train, Target_test):
    Data_train, Data_test = PCA(Data_train, Data_test)
    ml = OneVsRestClassifier(MLPClassifier(max_iter = 2000)).fit(Data_train, Target_train)
    return {"predict_proba": ml.predict_proba(Data_test),"predict": ml.predict(Data_test)}

def LBP_ML(Data_train, Data_test, Target_train, Target_test, imageShape):
    Data_train, Data_test = LBP(Data_train, Data_test, imageShape)
    ml = OneVsRestClassifier(MLPClassifier(max_iter = 2000)).fit(Data_train, Target_train)
    return {"predict_proba": ml.predict_proba(Data_test),"predict": ml.predict(Data_test)}

def LBP_SVM(Data_train, Data_test, Target_train, Target_test, imageShape):
    Data_train, Data_test = LBP(Data_train, Data_test, imageShape)
    svc = OneVsRestClassifier(SVC(probability=True, gamma = "auto")).fit(Data_train, Target_train)
    return {"predict_proba": svc.predict_proba(Data_test),"predict": svc.predict(Data_test)}

def LDA(Data_train, Data_test, Target_train, Target_test):
    lda = OneVsRestClassifier(LinearDiscriminantAnalysis()).fit(Data_train, Target_train)
    return {"predict_proba": lda.predict_proba(Data_test),"predict": lda.predict(Data_test)}

# --- Metricas
def EER(fpr, tpr, threshold):
    fnr = dict()
    eer_threshold = dict()
    eer = dict()
    for i in range(len(fpr)):
        fnr[i] = 1 - tpr[i]
        eer_threshold[i] = threshold[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]
        eer[i] = fpr[i][np.nanargmin(np.absolute((fnr[i] - fpr[i])))]
    return (eer, eer_threshold)

def ROC(classifier, Target_test, predict_Proba):
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()
    for i in range(len(Target_test[0])):
        fpr[i], tpr[i], threshold[i] = roc_curve(Target_test[:, i], predict_Proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(Target_test[0]))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(Target_test[0])):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= len(Target_test[0])
    
    # Plot all ROC curves
    lw = 2
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue'])
    for i, color in zip(range(len(Target_test[0])), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for the classifier '+classifier)
    plt.legend(loc="lower right")
    plt.show()
    
    return (fpr, tpr, threshold, roc_auc)

def ConfusionMatrix(Target_test, predict): #https://pythonhosted.org/bob/temp/bob.measure/doc/py_api.html
    return confusion_matrix(Target_test.dot(list(range(len(Target_test[0])))), predict.dot(list(range(len(Target_test[0])))), labels=range(len(Target_test[0])))

def CurvasPrecisaoRevocacao(classifier, Target_test, predict_Proba):
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(Target_test[0])):
        precision[i], recall[i], _ = precision_recall_curve(Target_test[:, i], predict_Proba[:, i])
        average_precision[i] = average_precision_score(Target_test[:, i], predict_Proba[:, i])
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    for i, color in zip(range(len(Target_test[0])), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to '+classifier)
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    
    
    plt.show()

def FMeasure(Target_test, Target_pred):
    return f1_score(Target_test.dot(list(range(len(Target_test[0])))), Target_pred.dot(list(range(len(Target_pred[0])))), average=None)  

# --- Teste
def test(Data_train, Data_test, Target_train, Target_test, imageShape):   
    # --- Extratores de caracteristicas e Classificadores
    
    predicts = {"LDA": LDA(Data_train, Data_test, Target_train, Target_test), 
                "PCA com Machine Learning": PCA_ML(Data_train, Data_test, Target_train, Target_test),
                "LBP com Machine Learning": LBP_ML(Data_train, Data_test, Target_train, Target_test, imageShape), 
                "LBP com Support Vector Machine": LBP_SVM(Data_train, Data_test, Target_train, Target_test, imageShape)}

    # --- Metricas
    roc_auc = dict()
    eer = dict()
    F1 = dict()
    PRC = dict()
    CM = dict()
    for key in predicts.keys():
        
        print("Dados do classificador: "+key)
        #
        CM[key] = ConfusionMatrix(Target_test,predicts[key]["predict"])
        
        #ROC
        fpr, tpr, threshold, roc_auc[key] = ROC(key, Target_test, predicts[key]["predict_proba"]) 
        
        #EER
        eer[key], eer_threshold = EER(fpr, tpr, threshold)
        
        #CurvasPrecisaoRevocacao
        PRC[key] = CurvasPrecisaoRevocacao(key, Target_test, predicts[key]["predict_proba"])
        
        #FMeasure
        F1[key] = FMeasure(Target_test, predicts[key]["predict"])
    
    return (predicts, CM, roc_auc, eer, PRC, F1)
    
#Data_train, Data_test, Target_train, Target_test, imageShape = getDatas(min_faces_per_person = 120, test_size = 0.25, saveDataBase = True)

predicts, CM, roc_auc, eer, PRC, F1 = test(Data_train, Data_test, Target_train, Target_test, imageShape)
'''
for i in range(len(Data_test)):
    Data_test[i].reshape(imageShape)[int((imageShape[0]-10)/2):int((imageShape[0]+10)/2)] = 0

Teste2 = test(Data_train, Data_test, Target_train, Target_test, imageShape)

for i in range(len(Data_test)):
    Data_test[i].reshape(imageShape)[int((imageShape[0]-10)/2):int((imageShape[0]+10)/2)] = 255

Teste3 = test(Data_train, Data_test, Target_train, Target_test, imageShape)
'''
