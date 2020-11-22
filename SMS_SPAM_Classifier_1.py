# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:03:44 2020

@author: Sanket Kotkar

SMS Spam Classifier
"""

#%% Importing the libraries 

import pandas as pd
import numpy as np
#%% 
Data = pd.read_csv("D:/Projects/SMS SPAM CLassification/Data/SMSSpamCollection", sep = '\t',
                   names = ['label','message'])

#%%   Data preprocessing and cleaning
import re
import nltk
nltk.download('stopwards')
#%%   

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []
for i in range(0,Data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ',Data['message'][i] )
    review = review.lower()
    review = review.split()
    
    review = [ ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#%%  Creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()
X = CV.fit_transform(corpus).toarray()
X = pd.DataFrame(X)
#%% 
Y = pd.get_dummies(Data['label'])
Y = Y.iloc[:,1].values
#%%  
from sklearn.model_selection import train_test_split
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size = 0.2, random_state= 42)

#%%
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
#%%
Model = MultinomialNB()
Model.fit(X_Train.iloc[:,:6150],Y_Train)
#%%
Y_Pred = Model.predict(X_Test.iloc[:,:6150])
cm = confusion_matrix(Y_Test,Y_Pred)
print(cm)
Acc = accuracy_score(Y_Test,Y_Pred)
print(Acc)
#%% 
