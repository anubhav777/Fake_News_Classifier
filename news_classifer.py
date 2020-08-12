import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('train.csv')
data=df.copy()
data.dropna(inplace=True)
data.reset_index()
X =data.drop(['id','title','author','label'],axis=1)
cv=CountVectorizer(ngram_range=(1,3))
arr=[]

for i in range(len(X)):
    print(i)
    new_text=re.sub('[^a-zA-z]',' ',X.iloc[i].text)
    new_text=new_text.lower()
    new_text=new_text.split()
    new_text = [ps.stem(sentence) for sentence in new_text if sentence not in (stopwords.words('english'))]
    new_text = ','.join(new_text)
    arr.append(new_text)

X=cv.fit_transform(arr)
y=data['label']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.32)
mb=MultinomialNB()
mb.fit(X_train,y_train)
y_pred= mb.predict(X_test)
acc=accuracy_score(y_test,y_pred)
