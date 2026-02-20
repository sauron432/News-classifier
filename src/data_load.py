from src.config import *

import pandas as pd

def data_load():
    try:
        real_news = pd.read_csv(DATAPATH_TRUE)
        fake_news = pd.read_csv(DATAPATH_FAKE)
        fake_news['label'] = 0
        real_news['label'] = 1
        news_df = pd.concat([real_news[0:100],fake_news[0:100]], axis=0)
        news_df = news_df.drop(['title','subject','date'], axis=1)
        return news_df
    except Exception as e:
        return "ERROR: ", e