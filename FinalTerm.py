import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score,KFold,LeaveOneOut
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def fetchDataset(data_path):
    
    data_dir_list = os.listdir(data_path)

    img_data_list=[]
    labels_list=[]


    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
            #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_resize=cv2.resize(input_img,(224,224))
            img_data_list.append(input_img_resize)
            labels_list.append(dataset)


    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    img_data = img_data/255

    """img_data.shape
    plt.imshow(img_data[212])
    plt.axis("off")"""

    #label categorization
    Label_Cat=LabelEncoder().fit_transform(labels_list)

    #Shuffle the dataset
    x,y = shuffle(img_data,Label_Cat, random_state=2)
    
    return x,y


#loading the VGG-19 network pre-trained on imagenet dataset
base_model2 = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')



FeaturesExtracted={}
for index, layer in enumerate(base_model2.layers):
    if (layer.name in ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool','fc1']):
        FeaturesExtracted[layer.name]=layer.output
    
    #print(index, layer.name )

def feature_Extractor(Feature,baseModel,Data):
    
    output = Feature
    output = keras.layers.Flatten()(output)
    vgg_model = keras.models.Model(inputs=baseModel.input,outputs=output)


    vgg_model.trainable = False
    for layer in vgg_model.layers:
        layer.trainable = False
        
    pd.set_option('max_colwidth', -1)
    layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]
    pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])  
    
    features_vgg = vgg_model.predict(Data, verbose=0)
    #validation_features_vgg = vgg_model.predict(TestData, verbose=0)

    print('Train Bottleneck Features:', features_vgg.shape)
    return features_vgg
    

#PCA transformation for dimensionality reduction

def getPCATrainTestData(Features,y,NPCA):
    
    PCAFeatures=[]

    
    pca = PCA( n_components = NPCA) 
    for feat in Features:
        X2D = pca.fit_transform(feat)
        #X2D=np.reshape(X2D,(28900,1))
        PCAFeatures.append(X2D)
    
    cutoff= int(len(PCAFeatures[0])*0.8)   
    


    X_train=[feature[:cutoff] for feature in PCAFeatures]
    y_train=y[:cutoff]
    X_test=[feature[cutoff:] for feature in PCAFeatures]
    y_test=y[cutoff:]
    
    return X_train, y_train, X_test, y_test


#classification function
def SVM_Classification(X_train, y_train, X_test, y_test):
    
    
    AccuracyScores=[]
    foldValidationScores=[]
    LOOValidationScores=[]
    
    for i in range(5):
        print("------------------------------------------")
        print("training for block pool {}".format(i+1))
        
        #SVM classifier
        clf=SVC(kernel='linear')
        clf.fit(X_train[i],y_train)
        
        #prediction
        prediction=clf.predict(X_test[i])
        
        #10-fold validation
        cv = KFold(n_splits=10, shuffle=True, random_state=1)
        fold_score = cross_val_score(clf, X_train[i], y_train, scoring='accuracy', cv=cv, n_jobs=-1)
        fold_score=fold_score.mean()
        print("10-Fold Cross Validation Scores:{}".format(fold_score))    

        #loocv = LeaveOneOut()
        looscores = cross_val_score(clf , X = X_train[i] , y =y_train ,scoring='accuracy', cv = LeaveOneOut())
        looscores=looscores.mean()
        print("Leave-One-Out cross Validation Scores:{}".format(looscores))

        #Test Accuracy
        acc=accuracy_score(y_test, prediction)
        print("Test Accuracy:{}".format(acc))
        
        #acc,fold_score,looscores =SVM_Classification(X_train[i],y_train,X_test[i],y_test)
        
        AccuracyScores.append(acc)
        foldValidationScores.append(fold_score)
        LOOValidationScores.append(looscores)
    

    
    return  AccuracyScores,foldValidationScores,LOOValidationScores

    

#draw a bar graph for a given  score for all the blocks
def plotResults(nPCA,block1,block2,block3,block4,block5,title):
       
    ax=plt.subplot(111)
    ax.bar(nPCA-10,block1,width=5, label=" Block 1", color='b')
    ax.bar(nPCA-5,block2,width=5, label="Block 2", color='g')
    ax.bar(nPCA,block3,width=5, label="Block 3", color='r')
    ax.bar(nPCA+5,block4,width=5, label="Block 4", color='c')
    ax.bar(nPCA+10,block5,width=5, label="Block 5", color='m')

    ax.set_xticks(nPCA)
    ax.set_ylim(ymin=60)
    plt.legend() 
    plt.xlabel('PCA')
    plt.ylabel('Accuracy(%)')

    plt.title(title)

    plt.show()

#get all 3 scores for pca=[50,100,150,200] 
def FacialExpRecog(x,y,baseModel):
    
    features_vgg=[]
    for key in FeaturesExtracted:
        print("getting feature vector for the layer {}".format(key))
        feat=feature_Extractor(FeaturesExtracted[key], baseModel,x)
        features_vgg.append(feat)
    
    nPCA=np.asarray([50,100,150,200])
    acc_Sc=[]
    AllPCA_fold_SCORES=[]
    AllPCA_loo_SCORES=[]

    for npca in nPCA:
        print("------------------------------------------")
        print("------------------------------------------")
        print("WHEN NPCA={}".format(npca))
        
        #getthing the PCA values
        X_train, y_train, X_test, y_test=getPCATrainTestData(features_vgg,y,npca)
        
        #Training the dataset
        AccuracyScores,foldValidationScores,LOOValidationScores=SVM_Classification(X_train, y_train, X_test, y_test)
        
        acc_Sc.append(AccuracyScores)
        AllPCA_fold_SCORES.append(foldValidationScores)
        AllPCA_loo_SCORES.append(LOOValidationScores)
        
    #plotting the accuracies
    b1=[scores[0]*100 for scores in acc_Sc]
    b2=[scores[1]*100 for scores in acc_Sc]
    b3=[scores[2]*100 for scores in acc_Sc]
    b4=[scores[3]*100 for scores in acc_Sc]
    b5=[scores[4]*100 for scores in acc_Sc]
    plotResults(nPCA,b1,b2,b3,b4,b5,"Test Accuracy")
    
    #plotting the 10-fold
    b1=[scores[0]*100 for scores in AllPCA_fold_SCORES]
    b2=[scores[1]*100 for scores in AllPCA_fold_SCORES]
    b3=[scores[2]*100 for scores in AllPCA_fold_SCORES]
    b4=[scores[3]*100 for scores in AllPCA_fold_SCORES]
    b5=[scores[4]*100 for scores in AllPCA_fold_SCORES]
    plotResults(nPCA,b1,b2,b3,b4,b5,"10-Fold Validation Accuracy")
    
    #plotting the accuracies
    b1=[scores[0]*100 for scores in AllPCA_loo_SCORES]
    b2=[scores[1]*100 for scores in AllPCA_loo_SCORES]
    b3=[scores[2]*100 for scores in AllPCA_loo_SCORES]
    b4=[scores[3]*100 for scores in AllPCA_loo_SCORES]
    b5=[scores[4]*100 for scores in AllPCA_loo_SCORES]
    plotResults(nPCA,b1,b2,b3,b4,b5,"Leave one out Cross Validation Accuracy")

        


    


#fetching JAFFE dataset
print("loading the  JAFFE dataset")
x,y=fetchDataset(data_path = './jaffe/')

#Training the JAFFE dataset
FacialExpRecog(x,y,base_model2)




#getting CK+ dataset
print("loading a subset of the  CK+ dataset")
x_ck,y_ck=fetchDataset(data_path = './CKPlus/')
#x_ck,y_ck=fetchDataset(data_path = './CK+48/')

#Training the CK+ dataset
FacialExpRecog(x_ck,y_ck,base_model2)