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
from gensim.models.word2vec import Word2Vec
import pickle


'''
需要先执行以上命令下载nltk包
'''


# int
def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


# 特殊的单个符号
def is_symbol(inputString):
    return bool(re.match(r'[^\w]', inputString))


def check(word):
    # 英文停用词
    stop = stopwords.words('english')
    word = word.lower()
    if word in stop:
        return False
    elif has_numbers(word) or is_symbol(word):
        return False
    else:
        return True


def clean_data(sen):
    res = []
    wordnet_lemmatizer = WordNetLemmatizer()
    for word in sen:
        if check(word):
            # 这一段的用处仅仅是去除python里面byte存str时候留下的标识
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res


def pickle_data(obj, file):

    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def reload_pickle(file):

    with open(file, 'rb')as f:
        data = pickle.load(f)
    return data


def pre_data(path='./data/Combined_News_DJIA.csv'):
    data = pd.read_csv(path)
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']

    X_train = train[train.columns[2:]]  # 句子从第二列开始 1611行、25列的表
    corpus = X_train.values.flatten().astype(str)  # 1611*25条句子，每一个元素就是一条新闻，长度不固定

    X_train = X_train.values.astype(str)
    X_train = np.array([' '.join(x) for x in X_train])  # 每一个元素就是25条句子，也就是当天的25条新闻，一共1611行

    X_test = test[test.columns[2:]]
    X_test = X_test.values.astype(str)
    X_test = np.array([' '.join(x) for x in X_test])

    y_train = train['Label'].values  # 0表示指数没涨，1表示涨了
    y_test = test['Label'].values
    # tokenize 句子
    corpus = [word_tokenize(x) for x in corpus]
    X_train = [word_tokenize(x) for x in X_train]
    X_test = [word_tokenize(x) for x in X_test]
    # 去掉停用词、去掉特殊字符以及数字
    corpus = [clean_data(x) for x in corpus]
    X_train = [clean_data(x) for x in X_train]
    X_test = [clean_data(x) for x in X_test]

    pickle_data(corpus, 'data/corpus.pkl')
    pickle_data(X_train, 'data/x_train.pkl')
    pickle_data(X_test, 'data/x_test.pkl')
    pickle_data(y_train, 'data/y_train.pkl')
    pickle_data(y_test, 'data/y_test.pkl')

    return corpus, X_train, y_train, X_test, y_test


def train():
    corpus = reload_pickle('data/corpus.pkl')
    X_train = reload_pickle('data/x_train.pkl')
    X_test = reload_pickle('data/x_test.pkl')

    # corpus: 训练集，可以是list
    # size: 输出的词向量的维度
    # window：表示当前词与预测词在一个句子中的最大距离是多少
    # min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉
    # workers: 并行处理数
    # 这里的vec是基于每个单词的
    model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=8)
    vocab = model.wv.vocab
    wordlist_train = X_train
    wordlist_test = X_test

    # 得到任意text的平均词向量的值
    def get_vector(word_list):
        # 建立一个全是0的array
        res = np.zeros([128])
        count = 0
        for word in word_list:
            if word in vocab:
                res += model[word]  # 取出单词，将每个单词的128维向量相加
                count += 1
        return res / count   # 求取所有词向量和的平均值

    X_train = [get_vector(x) for x in X_train]
    X_test = [get_vector(x) for x in X_test]

    return X_train, X_test


if __name__ == '__main__':
    # a = pre_data()

    b = train()
    print(b)
