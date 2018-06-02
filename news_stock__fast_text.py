from news_stock_predict import reload_pickle
import numpy as np
from sklearn import metrics
# import fasttext  # 这个库只能在mac和linux上运行


def save_data(data, file):

    with open(file, 'w')as f:
        for i in data:
            if isinstance(i, int):
                i = str(i)
            f.write(i+'\n')


def load_data():
    x_train = reload_pickle('data/x_train.pkl')
    x_test = reload_pickle('data/x_test.pkl')

    y_train = reload_pickle('data/y_train.pkl')
    y_test = reload_pickle('data/y_test.pkl')

    # 在x_train中加入lable
    for i in range(len(y_train)):
        label = '__label__' + str(y_train[i])
        x_train[i].append(label)

    x_train = [' '.join(x) for x in x_train]
    x_test = [' '.join(x) for x in x_test]

    # 需要三个东西：含有label的train， 不含label的test， 单独的label文件
    save_data(x_train, 'data/train_ft.txt')
    save_data(x_test, 'data/test_ft.txt')
    save_data(y_test, 'data/label_ft.txt')

    return x_train, y_train, x_test, y_test


def train(data):

    y_test = data[3]
    clf = fasttext.supervised('data/train_ft.txt', 'model', dim=256,
                              ws=5, neg=5, epoch=100, min_count=10, lr=0.1,
                              lr_update_rate=1000, bucket=200000)

    # 我们用predict来给出判断
    labels = clf.predict(X_test)

    y_preds = np.array(labels).flatten().astype(int)

    # 我们来看看
    print(len(y_test))
    print(y_test)
    print(len(y_preds))
    print(y_preds)

    # AUC准确率
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds, pos_label=1)
    print(metrics.auc(fpr, tpr))


if __name__ == '__main__':
    data = load_data()
    train(data)
