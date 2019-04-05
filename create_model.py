import pandas as pd
import re

df = pd.read_csv("labeled_data.csv", usecols=['class', 'tweet'])
df['tweet'] = df['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))

# 0 - hate speech 1 - offensive language 2 - neither

########################

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from stop_words import get_stop_words

clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)

clf = clf.fit(X=df['tweet'], y=df['class'])

########################

text = "I hate you, please die!"
clf.predict_proba([text])[0]

########################

from sklearn import externals

model_filename = "hatespeech.joblib.z"
externals.joblib.dump((clf), model_filename)
