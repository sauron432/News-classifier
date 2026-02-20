import pandas as pd
import spacy 

from src.data_preprocess import preprocess

def preprocess_dataframe(df:pd.DataFrame):
    nlp = spacy.load('en_core_web_lg')
    df_copy = df.copy()
    df_copy['processed_text'] = df_copy['text'].apply(preprocess)
    df_copy['vector'] = df_copy['processed_text'].apply(lambda x: nlp(x).vector)
    return df_copy
