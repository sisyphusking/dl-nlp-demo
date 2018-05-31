import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
'''
# 需要先执行以上命令下载nltk包
'''


# int
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


# 特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))


def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    stop = stopwords.words('english')
    word = word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True


def process(sen):
    res = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in sen:
        if check(word):
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识。。之前数据没处理好，其他case里不会有这个情况
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


def load_data(path='./data/Combined_News_DJIA.csv'):
    data = pd.read_csv(path)
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']

    X_train = train[train.columns[2:]]  # 句子从第二列开始  (1611, 25)维度的矩阵
    corpus = X_train.values.flatten().astype(str)  # 1611*25条句子

    X_train = X_train.values.astype(str)
    X_train = np.array([' '.join(x) for x in X_train])
    X_test = test[test.columns[2:]]
    X_test = X_test.values.astype(str)
    X_test = np.array([' '.join(x) for x in X_test])
    y_train = train['Label'].values  # 0表示指数没涨，1表示涨了
    y_test = test['Label'].values

    corpus = [word_tokenize(x) for x in corpus]
    X_train = [word_tokenize(x) for x in X_train]
    X_test = [word_tokenize(x) for x in X_test]

    corpus = [process(x) for x in corpus]
    X_train = [process(x) for x in X_train]
    X_test = [process(x) for x in X_test]

    return corpus, X_train, y_train, X_test, y_test


if __name__ == '__main__':
    a = load_data()
    print(a[0])

