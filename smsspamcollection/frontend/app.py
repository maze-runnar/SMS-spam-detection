import requests
from flask import Flask, request,render_template
import sys
import nltk
import sklearn
import pandas
import numpy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#import pickle ##load pre-trained model using pickle..
from nltk.tokenize import word_tokenize

from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier#SVM classsifier (support vector machine)
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



app = Flask(__name__)
app.config['DEBUG'] = True

df = pd.read_table('/home/saurabh/Desktop/smsspamcollection/SMSSpamCollection', header=None, encoding='utf-8') 

classes = df[0]

#from sklearn.preprocessing import LabelEncoder
# so convert spam to 1 and ham tabso 0
encoder = LabelEncoder()
y = encoder.fit_transform(classes)

text_messages = df[1]
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress')
# Replace URLs with 'webaddress'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                  'webaddress')
# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr')
# Replace numbers with 'numbr'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
# Remove punctuation
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')


processed = processed.str.lower()
from nltk.corpus import stopwords
s = stopwords.words('english')

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in s))


ps = nltk.PorterStemmer() # it removes the synonyms and similar sounding words..

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())



def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

from sklearn import svm
from joblib import dump, load


clf = load('/home/saurabh/Desktop/smsspamcollection/frontend/model1.joblib') 

@app.route('/', methods = ["GET","POST"])
def index():
	if(request.method == "POST"):
		#city = request.form['city']
		text_msg = request.form['sms']
		my_msg = find_features(text_msg)
		prediction =  clf.classify_many(my_msg)
		x = ""
		if(prediction[0] == 0):
			x = "Not a spam, it's ok "
		else:
			x = "it's a spam" 
		return render_template('mainpage.html',prediction = x)
	else:
		return render_template('mainpage.html')	


@app.route('/aboutme')
def about():
	return render_template('aboutme.html')



if __name__ == "__main__":
	app.run(debug = True)
