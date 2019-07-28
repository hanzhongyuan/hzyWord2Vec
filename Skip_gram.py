# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------+
#这是一个教学用的Skip-gram的实现
#
#Zhongyuan Han
#2019-7-26
#------------------------------------------------------------------------------+
import numpy as np
from collections import defaultdict

#1111111111111111111111111111111111111111111111111111111111111111111111111111111111111
#1.数据准备——定义语料库、整理、规范化和分词
text = 'natural language processing and machine learning is fun and exciting'

#一行一个句子，每个句子由若干个词组成
class Preprocess():  
    @staticmethod    
    def get_corpus( filename):
        corpus=[]
        file=open(filename)
        oneLine=file.readline();
        word_vector=oneLine.split( )
        corpus.append(word_vector)
        file.close()
        return corpus
    pass

preprocess =Preprocess()
corpus =preprocess.get_corpus("D:\\temp\\a.txt")
#[['natural', 'language', 'processing', 'and','machine','learning','is','fun','and','exciting'],
#    ['natural', 'language', 'processing', 'and','machine','learning','is','fun','and','exciting']]
#1111111111111111111111111111111111111111111111111111111111111111111111111111111111111


#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222
#2.定义超参数
parameters = {} #定义了字典，存放学习率、训练次数、窗口尺寸、嵌入（embedding）维度等参数
parameters['n'] = 5                   # 定义embedding的维度
parameters['window_size'] = 2         # 定义窗口大小
parameters['min_count'] = 0           # 最小单词出现次数
parameters['epochs'] = 50           # 训练次数
parameters['neg_sample'] = 10           # 反例数量
parameters['learning_rate'] = 0.01    # 学习速率
np.random.seed(0)                     # 定义随机数种子，随机数是固定顺序的，以保证实验可以重现
#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222

#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333
class Train_data():
    '''
    word_count #每个单词出现的次数，字典
    vocabularySize #不重复的单词个数，整数
    wordList #按字母排序的单词列表
    word_index #单词到编号的映射字典
    index_word #编号到单词的映射字典
    
    '''
   #初始化
    def __init__(self,corpus):
        self.word_count = self.get_word_count(corpus)
        self.vocabularySize = len(self.word_count.keys())
        self.wordList = sorted(list(self.word_count.keys()),reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.wordList))
        self.index_word = dict((i,word) for i, word in enumerate(self.wordList))
        
   
   #计算单词出现的次数
    def get_word_count(self,corpus):
        word_count = defaultdict(int)
        for oneline in corpus:
            for word in oneline:
                word_count[word]+=1
        return word_count

    #将给定的词转换成onehot向量
    def word2onehot(self,word):
        word_vec = [0 for i in range(0, self.vocabularySize)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    #生成训练数据
    def generate_training_data(self,settings,corpus):
       
        training_data = []
        # 读取语料中的每一个句子
        for sentence in corpus:            
            # 读取句子当中的每一个单词
            for i, word in enumerate(sentence):                
                #将当前单词转化为onehot向量
                w_target = self.word2onehot(sentence[i])
                # 平移窗口取内容
                w_context = []
                sentenceLength = len(sentence)
                for j in range(i - settings['window_size'], i + settings['window_size'] + 1):
                    if j != i and j <= sentenceLength - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)
#3生成训练数据
training_data = Train_data(corpus)
training_corpus = training_data.generate_training_data(parameters, corpus)
#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333

#4444444444444444444444444444444444444444444444444444444444444444444444444444444444444
class hzy_word2vec_Skip_gram():
    '''
    self.n # 定义embedding的维度
    self.epochs  # 训练次数
    self.learningRate  # 学习速率

    self.w1 #词向量转换矩阵
    self.w2 #隐层参数矩阵
    self.loss #损失函数
    '''

    #初始化参数
    def __init__(self,parameters):
        self.n = parameters['n']      # 定义embedding的维度
        self.epochs = parameters['epochs']  # 训练次数
        self.learningRate = parameters['learning_rate'] # 学习速率

    #定义softmax函数
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # 前向传递计算方法
    def forward_pass(self,x):
        h = np.dot(self.w1.T, x) #h=w1.*x
        u = np.dot(self.w2.T, h) #u=w2.*h
        y_c = self.softmax(u)    #y_c是u通过softmax归一化的结果
        return y_c, h, u

     # 反向传递算法
    def backprop(self,theError, h, x):
        dl_dw2 = np.outer(h, theError) 
        '''
       #np.outer 功能，
        将e展开成一列，然后用h里面每个数乘以e变成一列后的每一数，
        例如x1 = [[1,2],[3,4]]   x2 = [[1,1],[1,1]]   outer = np.outer(x1,x2)
        输出outer
        [[1 1 1 1]        #1倍
        [2 2 2 2]        #2倍
        [3 3 3 3]        #3倍
        [4 4 4 4]]       #4倍
        '''
        dl_dw1 = np.outer(x, np.dot(self.w2, theError.T))
        # 更新权重
        self.w1 = self.w1 - (self.learningRate * dl_dw1)
        self.w2 = self.w2 - (self.learningRate * dl_dw2)
        pass

    # 训练word2vector
    def train(self, training_data,training_corpus):
        # 初始化权重矩阵
        self.w1 = np.random.uniform(-0.8, 0.8, (training_data.vocabularySize, self.n))     #onthot到 embedding 的转换矩阵
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, training_data.vocabularySize)) # embedding到onthot的转换矩阵
        
        # 循环训练epochs次
        for i in range(0, self.epochs):
            self.loss = 0
            # 循环处理训练数据中的内容,word_taget是目标词，wordContext是目标词的上下文
            for wordTaget, wordContext in training_corpus:

                # 调用前向传递算法,#h=w1.*x #u=w2.*h #y_c是u通过softmax归一化的结果
                y_pred, h, u = self.forward_pass(wordTaget)
                
                # 计算错误率 np.subtract是求两者之差，然后求和
                theError = np.sum([np.subtract(y_pred, word) for word in wordContext], axis=0)

                # 调用反向传递算法进行更新
                self.backprop(theError, h, wordTaget)

                # 计算损失函数
                self.loss += -np.sum([u[word.index(1)] for word in wordContext]) + len(wordContext) * np.log(np.sum(np.exp(u)))
                #self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for
                #word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))                
            print 'EPOCH:',i, 'LOSS:', self.loss


    #输入一个onthot表示的词，得到他的词向量
    def get_word_vector(self,word,training_data):       
        w_index = training_data.word_index[word]
        v_w = self.w1[w_index]
        return v_w

     # 输入一个向量，得到最接近这个向量的词
    def vec_sim(self, vec, top_n):

        # 循环词汇表里的每个向量
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda (word, sim):sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print word, sim
            
        pass

#4 初始化
hzyWord2vector = hzy_word2vec_Skip_gram(parameters)


#4444444444444444444444444444444444444444444444444444444444444444444444444444444444444


#5555555555555555555555555555555555555555555555555555555555555555555555555555555555555
#训练模型
hzyWord2vector.train(training_data,training_corpus)
#5555555555555555555555555555555555555555555555555555555555555555555555555555555555555


#6666666666666666666666666666666666666666666666666666666666666666666666666666666666666
#6输出训练的向量
print("我们")
print(hzyWord2vector.get_word_vector('我们',training_data))
print("我")
print(hzyWord2vector.get_word_vector('我',training_data))

#6666666666666666666666666666666666666666666666666666666666666666666666666666666666666


   

     

      




'''
class hzy_word2vec_Skip_gram():

  

    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda (word, sim):sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print word, sim
            
        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):
        
        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda (word, sim):sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print word, sim
            
        pass

#--- EXAMPLE RUN
#--------------------------------------------------------------+
np.random.seed(0)                   # set the seed for reproducibility
corpus = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]

# INITIALIZE W2V MODEL
w2v = word2vec()

# generate training data
training_data = w2v.generate_training_data(settings, corpus)

# train word2vec model
w2v.train(training_data)
'''

#--- END ----------------------------------------------------------------------+