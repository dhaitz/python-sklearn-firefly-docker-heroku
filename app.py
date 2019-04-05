from sklearn import externals

model_filename = "hatespeech.joblib.z"
clf = externals.joblib.load(model_filename)

def predict(text):
    probas = clf.predict_proba([text])[0]
    return {'hate speech': probas[0], 'offensive language': probas[1], 'neither': probas[2]}
