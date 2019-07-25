#------------------------------------------------------------------------------+
#这是一个教学用的Skip-gram的实现
#
#Zhongyuan Han
#2019-7-26
#------------------------------------------------------------------------------+
import numpy as np
#import re
#from collections import defaultdict

#1111111111111111111111111111111111111111111111111111111111111111111111111111111111111
#1.数据准备——定义语料库、整理、规范化和分词
text = 'natural language processing and machine learning is fun and exciting'
coupus = [['natural', 'language', 'processing', 'and','machine','learning','is','fun','and','exciting']]
#1111111111111111111111111111111111111111111111111111111111111111111111111111111111111


#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222
#2.定义超参数
parameters = {} #定义了字典，存放学习率、训练次数、窗口尺寸、嵌入（embedding）维度等参数
parameters['n'] = 5                   # 定义embedding的维度
parameters['window_size'] = 2         # 定义窗口大小
parameters['min_count'] = 0           # 最小单词出现次数
parameters['epochs'] = 5000           # 训练次数
parameters['neg_sample'] = 10           # 反例数量
parameters['learning_rate'] = 0.01    # 学习速率
np.random.seed(0)                     # 定义随机数种子，随机数是固定顺序的，以保证实验可以重现
#2222222222222222222222222222222222222222222222222222222222222222222222222222222222222

#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333
#3生成训练数据
class Train_data():
    word_count #每个单词出现的次数，字典
    numberOfWord #不重复的单词个数，整数
    wordList #按字母排序的单词列表
    word_index #单词到编号的映射字典
    index_word #编号到单词的映射字典

    #计算单词出现的次数
    def get_word_count(corpus):
        words_count = defaultdict(int)
        for oneline in corpus:
            for word in oneline:
                word_count[word]+=1
        return words_count

    #生成训练数据
    def generate_training_data(settings, corpus):
        self.word_count = get_word_count(corpus)
        self.numberOfWord = len(word_count.keys())
        wordList = sorted(list(word_count.keys()),reverse=False)
        word_index = dict((word, i) for i, word in enumerate(wordList))
        word_index = dict((i,word) for i, word in enumerate(wordList))
        training_data = []
        # 读取语料中的每一个句子
        for sentence in corpus:
            sent_len = len(sentence)

            # 读取句子当中的每一个单词
            for i, word in enumerate(sentence):
                
                #w_target = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)
#3333333333333333333333333333333333333333333333333333333333333333333333333333333333333

#3 初始化
hzyWord2vector = hzy_word2vec_Skip_gram()

#4生成训练数据
training_data = hzyWord2vector.generate_training_data(settings, corpus)






   

     

      


# train word2vec model
w2v.train(training_data)



class hzy_word2vec_Skip_gram():

    
    def __init__(self):
        self.rate = settings['learning_rate']

        self.n = settings['n']
        self.eta = self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass
    
    
   


    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec


    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u
                

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)  
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass


    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))     # embedding matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))     # context matrix
        
        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:

                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)
                
                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                #self.loss += -2*np.log(len(w_c)) -np.sum([u[word.index(1)] for
                #word in w_c]) + (len(w_c) * np.log(np.sum(np.exp(u))))
                
            print 'EPOCH:',i, 'LOSS:', self.loss
        pass


    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w


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

#--- END ----------------------------------------------------------------------+