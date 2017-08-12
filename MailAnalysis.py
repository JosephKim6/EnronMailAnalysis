# encoding=utf8

import time
import json
import string
import collections
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib


# 向量化读取到的json文本
def vectorize(mail_list):

    # 提取邮件中的主题和文本
    mail_texts = [mail['Subject'] + " " +
                  reduce(lambda x, y: x + y, [part['content'] for part in mail['parts']]) for mail in mail_list]

    # 预处理
    preprocess_texts = []
    for text in mail_texts:
        # 小写化
        text = text.lower()
        # 标点变成空格
        for c in string.punctuation:
            text = text.replace(c, ' ')
        # 分词
        word_list = nltk.word_tokenize(text)
        # 去除停用词
        stopwords_list = stopwords.words('english')
        stopwords_list.extend(['http', 'com', 'www', 'pdf', 'html', 'org', 'phtml', 'asp'])
        filter_stopword_list = [w for w in word_list if w not in stopwords_list]
        # 词形还原
        lemmatizer = WordNetLemmatizer()
        lemmatized_list = [lemmatizer.lemmatize(w) for w in filter_stopword_list]
        # 留下词性为名词的单词
        postag_list = nltk.pos_tag(lemmatized_list)
        final_list = [w for w, pos in postag_list if pos == 'NN']
        # 预处理完成，加入list
        preprocess_texts.append(" ".join(final_list))

    # 把文本转换成词频向量
    tf_vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=2500)
    tf = tf_vectorizer.fit_transform(preprocess_texts)

    return tf, tf_vectorizer


# 训练LDA主题模型
def train_lda(tf, k):
    lda = LatentDirichletAllocation(n_topics=k, max_iter=100, learning_method='online')
    docres = lda.fit_transform(tf)

    return lda, docres


# 调参
def test_lda(tf, k):
    lda = LatentDirichletAllocation(n_topics=k, max_iter=20, learning_method='online')
    lda.fit(tf)

    return lda.perplexity(tf)


# 结果展示
def show_result(model, feature_names):
    for idx, topic in enumerate(model.components_):
        print "Topic #%d" % idx
        print " ".join([feature_names[i] for i in topic.argsort()[:-20 - 1:-1]])


# 把结果存入文件中
def save_result(model, feature_names, docre, mails_list, path, mails_path):
    f = open(path, 'w')
    for idx, topic in enumerate(model.components_):
        f.write("Topic #%d\n" % idx)
        f.write(" ".join([feature_names[i] for i in topic.argsort()[:-20 - 1:-1]]) + "\n")
    f.close()

    for idx, doc in enumerate(docre):
        topic = np.argmax(doc)
        mails_list[idx]['topic'] = topic
    joblib.dump(mails_list, mails_path)


if __name__ == '__main__':

    # 计算LDA，并保存
    date = '2001-05'
    mails = joblib.load('data/' + date + '.pkl')
    matrix, tf_model = vectorize(mails)
    lda_model, doc_result = train_lda(matrix, 9)
    save_result(lda_model, tf_model.get_feature_names(),
                doc_result, mails, 'result/' + date + '.txt', 'result/' + date + '-mails.pkl')

    # LDA调参
#    mails = joblib.load('data/2001-09.pkl')
#    matrix, tf_model = vectorize(mails)
#    n_topics = range(2, 10, 2)
#    perplexityLst = [1.0] * len(n_topics)
#    for idx, n_topic in enumerate(n_topics):
#        t0 = time.time()
#        perplexity = test_lda(matrix, n_topic)
#        perplexityLst[idx] = perplexity
#        print "# of Topic: %d, " % n_topics[idx],
#        print "done in %0.3fs, " % (time.time() - t0),
#        print "Perplexity Score %0.3f" % perplexityLst[idx]

    # 对所有邮件进行聚类处理
#    mails = [json.loads(line) for line in open('enron_20150507_inbox.mbox.json')]
#    matrix, tf_model = vectorize(mails)
#    lda_model = train_lda(matrix)
#    show_result(lda_model, tf_model.get_feature_names())

    # 把邮件按年月分类
#    date_mail = collections.defaultdict(list)
#    for mail in mails:
#        t = mail["Date"]["$date"] / 1000
#        localtime = time.localtime(t)
#        str_time = time.strftime('%Y-%m', localtime)
#        date_mail[str_time].append(mail)
#    for one in date_mail.items():
#        if len(one[1]) > 2000:
#            joblib.dump(one[1], "data/" + one[0] + ".pkl")
