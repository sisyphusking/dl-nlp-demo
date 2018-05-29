import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date


def load_data(path='./data/Combined_News_DJIA.csv'):
    data = pd.read_csv(path)
    return data


if __name__ == '__main__':
    print(load_data())