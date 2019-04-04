from sklearn import externals

model_filename = "movies.joblib.z"
clf, multilabel_binarizer = externals.joblib.load(model_filename)

def predict(text):
    return multilabel_binarizer.inverse_transform(clf.predict([text]))[0]