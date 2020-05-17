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



app = Flask(__name__)
app.config['DEBUG'] = True
 



df = pd.read_table('/home/saurabh/Desktop/smsspamcollection/SMSSpamCollection', header=None, encoding='utf-8') # don't use latin-1
classes = df[0]
##print(classes.value_counts())
encoder = LabelEncoder()
y = encoder.fit_transform(classes)
text_messages = df[1]
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress')
# Replace URLs with 'webaddress'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress')
# Replace money symbols with 'moneysymb' (£ can by typed with ALT key + 156)
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'£|\$', 'moneysymb')
    
# Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr')
processed = processed.str.replace(r'\d+(\.\d+)?', 'numbr')
# Remove punctuation
# you can use any regex expression they are basically taken from the wikipedia

processed = processed.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed = processed.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed = processed.str.replace(r'^\s+|\s+?$', '')
#as HORse horse Horse are same SO conver are letters to lower case
processed = processed.str.lower()
nltk.download('stopwords')
from nltk.corpus import stopwords
s = stopwords.words('english')
processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in s))
# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer() # it removes the synonyms and similar sounding words..

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# create bag-of-words
all_words = []

for message in processed:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys()) #using all most common words as features to increase accuracy
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features
messages = zip(processed, y)

# define a seed for reproducibility
seed = 1
np.random.seed = seed
#np.random.shuffle(messages)

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]

from sklearn import model_selection

# split the data into training and testing datasets
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)

from nltk.classify.scikitlearn import SklearnClassifier
#SVM classsifier (support vector machine)
from sklearn.svm import SVC
model1 = SklearnClassifier(SVC(kernel = 'linear'))
model1.train(training)
#accuracy = nltk.classify.accuracy(model1, testing)

#import math
#print("SVC Classifier accuracy {}%".format(round(accuracy * 100,4)))


@app.route('/', methods = ["GET","POST"])
def index():
	if(request.method == "POST"):
		#city = request.form['city']
		text_msg = request.form['sms']
		my_msg = find_features(text_msg)
		prediction = model1.classify_many(my_msg)
		return render_template('mainpage.html',prediction = prediction)
	else:
		return render_template('mainpage.html')	


@app.route('/aboutme')
def about():
	return render_template('aboutme.html')



if __name__ == "__main__":
	app.run(debug = True)
