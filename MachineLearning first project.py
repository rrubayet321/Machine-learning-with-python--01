#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Please note that the more and cleaner the data is the more accuratetely it produces the result in MachineLearning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier #we used scikicklearn library to import decision tree algo so that our model learns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data= pd.read_csv('music.csv')
# music_data
X = music_data.drop(columns= ['genre']) #we use by convention X to determine input data set in ML
Y= music_data['genre'] #And we use Y  by convention to represent output data set in ML
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size= 0.2)


model= DecisionTreeClassifier()
model.fit(X_train,Y_train) #we used fit method to train our model by giving it the input and output dataset
predictions= model.predict(X_test) #we are asking our model to make predictions acccording to our given input and output

score= accuracy_score(Y_test,predictions) #this function returns accuracy score between 0 and 1
score


# In[12]:


#Since model training is very time consuming so we use a code that makes the model already trained
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

music_data= pd.read_csv('music.csv')
X = music_data.drop(columns= ['genre']) #we use by convention X to determine input data set in ML
Y= music_data['genre'] #And we use Y  by convention to represent output data set in ML

model=DecisionTreeClassifier()
model.fit(X,Y)

tree.export_graphviz(model, out_file= "music-recommender.dot",
                     feature_names=['age','gender'],
                     class_names= sorted(Y.unique()),
                     label= "all",
                     rounded= True,
                     filled= True
)
                    

