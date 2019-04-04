import pandas as pd

df = pd.read_csv('movies.csv')

df['title'] = df['title'].apply(lambda x: x.split('(')[0])
df = df[df['genres'] != '(no genres listed)']
df['genres'] = df['genres'].apply(lambda x: x.split('|'))

########################

from sklearn import preprocessing

mlb = preprocessing.MultiLabelBinarizer()
targets_mlb = mlb.fit_transform(df['genres'])

########################

from sklearn import pipeline, feature_extraction, multiclass, svm

clf = pipeline.make_pipeline(
    feature_extraction.text.TfidfVectorizer(),
    multiclass.OneVsRestClassifier(svm.LinearSVC())
)
clf = clf.fit(X=df['title'], y=targets_mlb)

########################

text = "Space Zombies"
mlb.inverse_transform(clf.predict([text]))[0]

########################

from sklearn import externals

model_filename = 'movies.joblib.z'
externals.joblib.dump((clf, mlb), model_filename)
