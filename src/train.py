from sklearn.linear_model import LogisticRegression
from pickle import dump


clf = LogisticRegression()
clf.fit(x_train, y_train)

with open("model.pkl", "wb") as f:
    dump(clf, f, protocol=5)