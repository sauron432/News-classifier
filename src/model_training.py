from sklearn.linear_model import LogisticRegression
import joblib

def train_model(X_train,y_train):
    LR_classifier = LogisticRegression()
    LR_classifier.fit(X_train,y_train)
    model = LR_classifier
    with open('model/LR_classifier.pkl','wb') as file:
        joblib.dump(model,file)
    print('Model saved as LR_classifier.pkl')
