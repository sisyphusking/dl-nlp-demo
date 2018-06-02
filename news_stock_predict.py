import pandas as pd
# import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from keras.models import Sequential
from keras.layers import *
from keras import backend as K
K.set_image_dim_ordering('tf')
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from gensim.models.word2vec import Word2Vec
import pickle

'''
需要先执行以上命令下载nltk包
'''

# cnn model parameters:
batch_size = 32
n_filter = 16
filter_length = 4
nb_epoch = 5
n_pool = 2


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


# 说明，对于每天的新闻，我们会考虑前256个单词。不够的我们用[000000]补上
# vec_size 指的是我们本身vector的size
def transform_to_matrix(model, x, padding_size=256, vec_size=128):
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                # 这里有两种except情况
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，我们直接贴上全是0的vec，最终要得到的是128*256的矩阵
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res


def load_data():
    corpus = reload_pickle('data/corpus.pkl')
    X_train = reload_pickle('data/x_train.pkl')
    X_test = reload_pickle('data/x_test.pkl')

    y_train = reload_pickle('data/y_train.pkl')
    y_test = reload_pickle('data/y_test.pkl')

    # corpus: 训练集，可以是list
    # size: 输出的词向量的维度
    # window：表示当前词与预测词在一个句子中的最大距离是多少
    # min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉
    # workers: 并行处理数
    # 这里的vec是基于每个单词的，将每一个单词都转化成128维度的向量
    model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=8)
    vocab = model.wv.vocab
    wordlist_train = X_train  # 复制一份原始的数据集
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

    # X_train以及X_test这种长度不一致的可以传入SVM等机器学习算法来处理
    X_train = [get_vector(x) for x in X_train]  # x长度不固定，都可以转化成128维度的向量
    X_test = [get_vector(x) for x in X_test]

    # 这里主要用到x_train和x_test，交给神经网络处理
    # 得到固定长度的矩阵， 将1611*n的矩阵，取n的前256个单词，不足的补0，每个单词是128维的词向量
    x_train = transform_to_matrix(model, wordlist_train)
    x_test = transform_to_matrix(model, wordlist_test)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    # 这里要reshape一下，让每一个matrix外部“包裹”一层维度。来告诉我们的CNN model，我们的每个数据点都是独立的。之间木有前后关系。
    # 请注意，这里是tensorflow作为后端，因此色阶应该在后面，输入维度应该为 (samples, height, width, channels)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    return x_train, y_train, x_test, y_test


def model():
    '''
    输入符合cnn的要求
    '''

    # 新建一个sequential的模型
    model = Sequential()
    model.add(Conv2D(n_filter, (filter_length, filter_length),
                            input_shape=(256, 128, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filter, (filter_length, filter_length)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    # 后面接上一个mlp，也可以用LSTM或者RNN等接在后面，可以提高精度
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))
    # compile模型
    model.compile(loss='mse',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    model.summary()
    return model


def train(model, data):

    model.fit(data[0], data[1], batch_size=batch_size, epochs=nb_epoch,
              verbose=0)
    model.save('model.h5')
    score = model.evaluate(data[2], data[3], verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    # a = pre_data()  #  prepare data
    data_set = load_data()
    model = model()
    train(model, data_set)

