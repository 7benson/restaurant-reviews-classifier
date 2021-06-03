import os
from os import walk,path
from flask import Flask,render_template,request,redirect
import pandas as pd
import numpy as np
import joblib
import nltk
nltk.download('all')  
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

app=Flask(__name__)

model=joblib.load('./static/model/restaurent_review_model.pkl')
stemmer=PorterStemmer()
@app.route('/')
def home(message=""):
    redirect('/')
    return render_template('index.html',message=message)

@app.route('/predict/',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        print("\n\n\n\n\n\n\n")
        sentence=request.form.get('sentence')
        print('sentence===>',sentence)
        if(sentence==""):
            return home(message='Enter your Message')
        result=preprocess_and_predict(request.form.get('sentence'))
        print("\n\n\n\n\n\n\n")
        print("\n\n\n\n\n\n\n")
        if(result==1):
            return render_template('predict.html',text="Your Restaurent is sooo Good")
        else:
            return render_template('predict.html',text="Your Restaurent is n't good enough to visit")


def preprocess_and_predict(sentence):
    corpus=joblib.load('./static/variables_dump/corpus.pkl')
    sentence=re.sub('[^A-Za-z]',' ',sentence)
    sentence=sentence.lower()
    words=nltk.word_tokenize(sentence)
    words=[stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    sentence=' '.join(words)
    corpus.append(sentence)

    cv=CountVectorizer(ngram_range=(1,2),max_features=5000)
    corpus=cv.fit_transform(corpus).toarray()
    x_temp=corpus[-1]
    x_temp=np.array(x_temp).reshape((1,5000))
    return model.predict(x_temp)[0]

extra_dirs = ['./templates']
extra_files = extra_dirs[:]
for extra_dir in extra_dirs:
    for dirname, dirs, files in walk(extra_dir):
        for filename in files:
            filename = path.join(dirname, filename)
            if path.isfile(filename):
                extra_files.append(filename)

if __name__ == '__main__':
    app.run(debug=True,extra_files=extra_files)