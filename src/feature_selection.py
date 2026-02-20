import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def feature_selection(news_df:pd.DataFrame):
    try:
        input = news_df.vector
        output = news_df.label
        X_train, X_test, y_train, y_test = train_test_split(
            input,
            output,
            test_size=0.3,
            random_state=43,
            shuffle=True,
            stratify=output
        )
        X_train = np.stack(X_train)
        X_test = np.stack(X_test)
        return X_train,X_test,y_train,y_test
    
    except Exception as e:
        return "Error: ", e
    