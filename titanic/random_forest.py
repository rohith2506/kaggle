import numpy as np
import pandas as pd
import pdb
from sklearn.ensemble import RandomForestClassifier

def train_model():
    train_data = pd.read_csv("input/train.csv")
    test_data = pd.read_csv("input/test.csv")

    y = train_data["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv("submission.csv", index=False)

train_model()
