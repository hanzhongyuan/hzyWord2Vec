using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Word2VecCSharp
{
    //每个词的基本数据结构
    class vocab_word
    {
        public Int64 oneHot;
        public Int64 cn;//词频，从训练集中计数得到或直接提供词频文件        
        public string word;//word为该词的字面值        
    }


    class Hzyword2Vector
    {
        public static int MAX_STRING = 100; //词的最大长度
        public static int EXP_TABLE_SIZE = 1000;  //exp速查表的大小
        public static int MAX_EXP = 6;  //最大exp的值
        public static int MAX_SENTENCE_LENGTH = 1000; //句子的最大长度（词数)

        //词向量的长度
        public static int embeddinglength = 100;
        //词的信息
        public static ConcurrentDictionary<string, vocab_word> word_information = new ConcurrentDictionary<string, vocab_word>();
        //负反馈存储的词的分布
        public static List<string> table = new List<string>(); //table 按照词频分布存储单词
                                                               //syn0存储的是词表中每个词的词向量        
                                                               //syn1neg是负采样时每个词的辅助向量
                                                               //expTable是提前计算好的Sigmond函数表
        public static double[] syn0, syn1neg, expTable;
        public static int debug_mode = 2, window = 5, min_count = 5;

        public static Int64 train_words = 0;//总词数
        public static Int64 word_count_actual = 0, iter = 5;
        public static double alpha = 0.025, sample = 1e-3;
        public static int negative = 5;

        public static int starting_alpha;
        public Hzyword2Vector()
        {
            expTable = new double[EXP_TABLE_SIZE + 1];
            for (int i = 0; i < EXP_TABLE_SIZE; i++)
            {
                expTable[i] = Math.Exp((1.0 * i / (EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
                expTable[i] = expTable[i] / (expTable[i] + 1);
            }
        }

        static void Main(string[] args)
        {
            string trainfilename = @"D:\temp\zh_wiki_segment.txt";
            string outputfilename = @"D:\temp\zh_wiki_embedding.txt";
            Hzyword2Vector hzyword2Vector = new Hzyword2Vector();
            hzyword2Vector.LearnVocabFromTrainFile(trainfilename);
            hzyword2Vector.InitUnigramTable();
            hzyword2Vector.InitNet();
            hzyword2Vector.TrainModelThread(15, trainfilename);
            hzyword2Vector.saveEmbedding(outputfilename);
        }


        //计算每个函数的词频分布表，在负采样中用到
        //负采样算法：带权采样思想。每个词的权重为l(w) = [counter(w)]^(3/4) / sum([counter(u)]^(3/4))，u属于词典D
        //每个词对应一个线段, 将[0,1]等距离划分成10^8，每次生成一个随机整数r，Table[r]就是一个样本。
        public void InitUnigramTable()
        {
            double train_words_pow = 0;
            double d1;
            double power = 0.75;
            //遍历词表，根据词频计算累计值的分母
            foreach (var word in word_information)
            {
                train_words_pow += Math.Pow(word.Value.cn, power); //  求和( 词频^0.75)
            }
            foreach (var word in word_information)
            {
                d1 = Math.Pow(word.Value.cn, power) / train_words_pow; //  求和( 词频^0.75)
                for (double i = 0; i < d1; i = i + 1.0 / 1e8)
                {
                    table.Add(word.Key);
                }
            }
        }

        //为一个词构建一个vocab_word结构对象，并添加到词表中
        //词频初始化为0，hash值用之前的函数计算，
        //返回该词在词表中的位置
        void AddWordToVocab(string word)
        {
            if (word.Length > MAX_STRING)
            {
                word = word.Substring(0, MAX_STRING);
            }
            if (!word_information.ContainsKey(word))
            {
                vocab_word one_vocab_word = new vocab_word();
                one_vocab_word.cn = 1;
                one_vocab_word.word = word;
                if (!word_information.TryAdd(word, one_vocab_word))
                {
                    Console.WriteLine("插入单词失败");
                }
            }
            else
            {
                word_information[word].cn += 1;
            }

        }

        //统计词频，按照词频对词表中的项从大到小排序
        //统计总单词数量
        //这个排序是干什么用的？构建哈夫曼树？
        void SortVocab()
        {
            foreach (var word in word_information.Keys)
            {
                if (word_information[word].cn < min_count)
                {
                    vocab_word temp;
                    if (!word_information.TryRemove(word, out temp))
                    { Console.WriteLine("移除出现词数少的词失败"); }
                }
            }
            train_words = 0;
            foreach (var word in word_information.Values)
            {
                train_words += word.cn;
            }
            Console.WriteLine("还没有排序，onehotid还没有实现更新");
        }

        //从训练文件中获取所有词汇并构建词表和hash词表
        void LearnVocabFromTrainFile(string filename)
        {
            StreamReader file = new StreamReader(filename);
            while (!file.EndOfStream)
            {
                string[] fields = file.ReadLine().Split(' ');
                foreach (var word1 in fields)
                {
                    AddWordToVocab(word1);
                }
            }
            file.Close();

            //对词表进行排序，剔除词频低于阈值min_count的值，输出当前词表大小和总词数
            SortVocab();
            if (debug_mode > 0)
            {
                Console.WriteLine("Vocab size: " + word_information.Count);
                Console.WriteLine("Words in train file: " + train_words);
            }

        }


        //将单词和对应的词频输出到文件中
        void SaveVocab(string filename)
        {
            StreamWriter file = new StreamWriter(filename);
            foreach (var item in word_information)
            {
                file.WriteLine(item.Value.word + "\t" + item.Value.cn);
            }
            file.Close();
        }



        //从词汇表文件中读词并构建词表和hash表
        //由于词汇表中的词语不存在重复，因此与LearnVocabFromTrainFile相比没有做重复词汇的检测
        void ReadVocab(string filename)
        {
            word_information.Clear();
            StreamReader file = new StreamReader(filename);
            while (!file.EndOfStream)
            {
                string[] fields = file.ReadLine().Split('\t');
                vocab_word word1 = new vocab_word();
                word1.word = fields[0];
                word1.cn = Convert.ToInt64(fields[1]);
                if (!word_information.TryAdd(fields[0], word1)) { Console.WriteLine("添加单词失败"); }
            }
        }

        //初始化神经网络结构
        public void InitNet()
        {
            Int64 a, b;
            Random next_random = new Random();
            //syn0存储的是词表中每个词的词向量,词汇表大小 * 词向量的长度           
            syn0 = new double[word_information.Count * embeddinglength];

            //如果要使用负采样，则需要为syn1neg分配内存空间
            //syn1neg是负采样时每个词的辅助向量
            if (negative > 0)
            {
                syn1neg = new double[word_information.Count * embeddinglength];
                for (a = 0; a < word_information.Count; a++)
                {
                    for (b = 0; b < embeddinglength; b++)
                    {
                        syn1neg[a * embeddinglength + b] = 0;
                    }
                }
            }

            for (a = 0; a < word_information.Count; a++)
            {
                for (b = 0; b < embeddinglength; b++)
                {
                    //初始化词向量syn0，每一维的值为[-0.5, 0.5]/layer1_size范围内的随机数
                    syn0[a * b] = (next_random.NextDouble() - 0.5) / embeddinglength;
                }
            }

        }

        public void saveEmbedding(string filename)
        {
            StreamWriter file = new StreamWriter(filename);
            foreach (var word in word_information)
            {
                file.Write(word.Key + "\t");
                for (int i = 0; i < embeddinglength; i++)
                {
                    file.Write(syn0[word.Value.oneHot + i] + "\t");
                }
                file.WriteLine();
            }
            file.Close();
        }

        //该函数为线程函数，是训练算法代码实现的主要部分
        //默认在执行该线程函数前，已经完成词表排序、Haffman树的生成以及每个词的Haffman编码计算
        public void TrainModelThread(Int64 id, string train_file)
        {
            string last_word;
            int sentence_position = 0;
            Int64 word_count = 0, last_word_count = 0; //word_count: 当前线程当前时刻已训练的语料的长度//last_word_count: 当前线程上一次记录时已训练的语料长度   
            List<string> sen = new List<string>();  //sen：当前从文件中读取的待处理句子，存放的是每个词            
            Int64 l1, l2, local_iter = iter;//l1：在skip-gram模型中，在syn0中定位当前词词向量的起始位置 //l2：在syn1或syn1neg中定位中间节点向量或负采样向量的起始位置

            string target; //target：在负采样中存储当前样本
            int label;//label：在负采样中存储当前样本的标记

            Random next_random = new Random(); //next_random：用来辅助生成随机数
            double f, g;
            int currentPosition;
            double[] neu1 = new double[embeddinglength]; //neu1：输入词向量，在CBOW模型中是Context(x)中各个词的向量和，在skip-gram模型中是中心词的词向量
            double[] neu1e = new double[embeddinglength]; //neuele：累计误差项

            StreamReader file = new StreamReader(train_file);

            //开始主循环
            while (!file.EndOfStream)
            {
                //每训练约10000词输出一次训练进度
                if (word_count - last_word_count > 10000)
                {
                    //word_count_actual是所有线程总共当前处理的词数
                    word_count_actual += word_count - last_word_count;
                    last_word_count = word_count;
                    //在初始学习率的基础上，随着实际训练词数的上升，逐步降低当前学习率（自适应调整学习率）
                    alpha = starting_alpha * (1 - word_count_actual / (double)(iter * train_words + 1));
                    //调整的过程中保证学习率不低于starting_alpha * 0.0001
                    if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
                }

                //从训练样本中取出一个句子，句子间以回车分割                
                if (sen.Count == 0)
                {
                    string[] words = file.ReadLine().Split(' ');
                    foreach (var one_word in words)
                    {
                        if (word_information.ContainsKey(one_word))
                        {
                            //对高频词进行随机下采样，丢弃掉一些高频词，能够使低频词向量更加准确，同时加快训练速度//可以看作是一种平滑方法                        
                            if (sample > 0)
                            {
                                double ran = (Math.Sqrt(word_information[one_word].cn / (sample * train_words)) + 1) * (sample * train_words) / word_information[one_word].cn;
                                if (ran < next_random.NextDouble()) continue; //以1-ran的概率舍弃高频词
                            }
                            sen.Add(one_word);
                        }
                        //如果句子长度超出最大长度则截断
                        if (sen.Count >= MAX_SENTENCE_LENGTH) break;
                    }
                    //定位到句子头
                    sentence_position = 0;
                }

                //取出当前单词
                foreach (string word in sen)
                {

                    if (!word_information.ContainsKey(word)) continue;

                    //初始化输入词向量
                    for (int i = 0; i < embeddinglength; i++) neu1[i] = 0;
                    //初始化累计误差项
                    for (int i = 0; i < embeddinglength; i++) neu1e[i] = 0;

                    //生成一个[0, window-1]的随机数，用来确定|context(w)|窗口的实际宽度（提高训练速率？）
                    int ture_window_size = next_random.Next() % window;

                    //因为需要预测Context(w)中的每个词，因此需要循环2window - 2b + 1次遍历整个窗口//遍历时跳过中心单词                    
                    for (int leftWindowPosition = ture_window_size; leftWindowPosition < window * 2 + 1 - ture_window_size; leftWindowPosition++)
                    {
                        if (leftWindowPosition != window)
                        {// 假设window=5 ture_windows_size=3,lefitwindowsPostion的取值3，略过5，增加到7,即3467,currentPosition的取值就是句子当前位置-2，-1，+1，+2
                         // 假设window=5 ture_windows_size=1,lefitwindowsPostion的取值1，略过5，增加到9,即12346789,currentPosition的取值就是句子当前位置-4，-3，-2，-1，+1，+2，+3，+4
                            #region #取一个上下文的词
                            currentPosition = sentence_position - window + leftWindowPosition;
                            if (currentPosition < 0) continue;
                            if (currentPosition >= sen.Count) continue;
                            last_word = sen[currentPosition];//last_word为当前待预测的上下文单词
                            if (last_word == "") continue;
                            #endregion


                            //l1为当前答案单词的词向量在syn0中的起始位置
                            l1 = word_information[last_word].oneHot * embeddinglength;
                            //初始化累计误差
                            for (int i = 0; i < embeddinglength; i++) neu1e[i] = 0;


                            //如果采用负采样优化
                            //遍历所有正负样本（1个正样本+negative个负样本）
                            //算法流程基本和CBOW的ns一样，也采用的是模型对称
                            if (negative > 0)
                            {
                                for (int d = 0; d < negative + 1; d++)
                                {
                                    #region # d=0 取正样本 d=1..负样本数+1 取负样本
                                    if (d == 0)
                                    {
                                        target = word;
                                        label = 1;
                                    }
                                    else
                                    {
                                        target = table[next_random.Next() % table.Count]; //按照概率取负样本单词
                                        if (target == "") target = word_information[(table[next_random.Next() % (word_information.Count - 1) + 1])].word;
                                        if (target == word) continue;
                                        label = 0;
                                    }
                                    #endregion

                                    //target是正负样本词，是正样本还是负样本看标签label
                                    l2 = word_information[target].oneHot * embeddinglength;
                                    f = 0;
                                    for (int i = 0; i < embeddinglength; i++)
                                    {
                                        f += syn0[i + l1] * syn1neg[i + l2];   ////f为输入向量与辅助向量的内积    
                                    }
                                    if (f > MAX_EXP)
                                    {
                                        g = (label - 1) * alpha;
                                    }
                                    else
                                    {
                                        if (f < -MAX_EXP)
                                        { g = (label - 0) * alpha; }
                                        else
                                        {
                                            g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                                        }
                                    }
                                    //用辅助向量和g更新累计误差
                                    for (int i = 0; i < embeddinglength; i++)
                                    {
                                        neu1e[i] += g * syn1neg[i + l2];
                                    }
                                    //用输入向量和g更新辅助向量
                                    for (int i = 0; i < embeddinglength; i++)
                                    {
                                        syn1neg[i + l2] += g * syn0[i + l1];
                                    }
                                }
                                //更新词向量
                                for (int i = 0; i < embeddinglength; i++)
                                { syn0[i + l1] += neu1e[i]; }
                            }
                        }


                        //完成了一个词的训练，句子中位置往后移一个词
                        sentence_position++;
                        //处理完一句句子后，将句子长度置为零，进入循环，重新读取句子并进行逐词计算
                        if (sentence_position >= sen.Count)
                        {
                            sen = new List<string>();
                            continue;
                        }
                    }
                }
            }

            file.Close();
        }
    }
    class Hzyword2Vectorbak
    {
        public static int MAX_STRING = 100; //词的最大长度
        public static int EXP_TABLE_SIZE = 1000;  //exp速查表的大小
        public static int MAX_EXP = 6;  //最大exp的值
        public static int MAX_SENTENCE_LENGTH = 1000; //句子的最大长度（词数)
        public static int MAX_CODE_LENGTH = 40;

        public static ConcurrentDictionary<string, vocab_word> vocab = new ConcurrentDictionary<string, vocab_word>();

        public static int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

        //vocab_size为训练集中不同单词的个数，即词表的大小,这里直接用函数代替，为了和原来代码较为一致
        //layer1_size为词向量的长度
        public static int vocab_size() { return vocab.Count; }
        public static int layer1_size = 100;

        public static Int64 train_words = 0;//总词数
        public static Int64 word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

        public static double alpha = 0.025, starting_alpha, sample = 1e-3;

        //syn0存储的是词表中每个词的词向量
        //syn1存储的是Haffman树中每个非叶节点的向量
        //syn1neg是负采样时每个词的辅助向量
        //expTable是提前计算好的Sigmond函数表

        public static double[] syn0, syn1, syn1neg, expTable;
        public static int hs = 0, negative = 5;


        public static List<string> table; //table 存储的是词频分布表
        public static int table_size() { return table.Count; }
        static void Main1(string[] args)
        {


            Hzyword2Vector hzyword2Vector = new Hzyword2Vector();
            hzyword2Vector.InitUnigramTable();
            hzyword2Vector.InitNet();


            hzyword2Vector.TrainModelThread(15, @"d:\temp\word2vectortrain.txt");
        }


        //计算每个函数的词频分布表，在负采样中用到
        //负采样算法：带权采样思想。每个词的权重为l(w) = [counter(w)]^(3/4) / sum([counter(u)]^(3/4))，u属于词典D
        //每个词对应一个线段, 将[0,1]等距离划分成10^8，每次生成一个随机整数r，Table[r]就是一个样本。
        public void InitUnigramTable()
        {

            double train_words_pow = 0;
            double d1;
            double power = 0.75;

            //为词频分布表分配内存空间，
            table = new List<string>();

            //遍历词表，根据词频计算累计值的分母
            foreach (var word in vocab)
            {
                train_words_pow += Math.Pow(word.Value.cn, power); //  求和( 词频^0.75)
            }

            foreach (var word in vocab)
            {
                d1 = Math.Pow(word.Value.cn, power) / train_words_pow; //  求和( 词频^0.75)
                for (double i = 0; i < d1; i = i + 1.0 / 1e8)
                {
                    table.Add(word.Key);
                }
            }
        }

        /// <summary>
        /// //////////////
        /// 爱上对方拒绝了阿斯兰的看法就阿斯顿发链接啊多放辣椒士大夫立刻就啊士大夫立刻就士大夫
        /// </summary>
        /// <param name="fin"></param>
        /// <returns></returns>

        //从文件中读入一个词到word，以space' '，tab'\t'，EOL'\n'为词的分界符
        //截去一个词中长度超过MAX_STRING的部分
        //每一行的末尾输出一个</s>
        string ReadWord(StreamReader fin)
        {
            throw new Exception("方法未实现");
        }


        //返回一个词的hash值，由词的字面值计算得到，可能存在不同词拥有相同hash值的冲突情况
        int GetWordHash(string word)
        {
            throw new Exception("方法未实现");
        }

        //返回一个词在词表中的位置，若不存在则返回-1
        //先计算词的hash值，然后在词hash表中，以该值为下标，查看对应的值
        //如果为-1说明这个词不存在索引，即不存在在词表中，返回-1
        //如果该索引在词表中对应的词与正在查找的词不符，说明发生了hash值冲突，按照开放地址法去寻找这个词
        int SearchVocab(string word)
        {
            throw new Exception("方法未实现");
        }

        //从文件中读入一个词，并返回这个词在词表中的位置，相当于将之前的两个函数包装了起来
        int ReadWordIndex(StreamReader fin)
        {
            throw new Exception("方法未实现");
        }

        //为一个词构建一个vocab_word结构对象，并添加到词表中
        //词频初始化为0，hash值用之前的函数计算，
        //返回该词在词表中的位置
        void AddWordToVocab(string word)
        {
            if (word.Length > MAX_STRING)
            {
                word = word.Substring(0, MAX_STRING);
            }
            if (!vocab.ContainsKey(word))
            {
                vocab_word one_vocab_word = new vocab_word();
                one_vocab_word.cn = 1;
                one_vocab_word.word = word;
                if (!vocab.TryAdd(word, one_vocab_word))
                {
                    Console.WriteLine("插入单词失败");
                }
            }
            else
            {
                vocab[word].cn += 1;
            }

        }



        //输入两个词的结构体，返回二者的词频差值   
        Int64 VocabCompare(vocab_word a, vocab_word b)
        {
            return b.cn - a.cn;
        }




        //统计词频，按照词频对词表中的项从大到小排序
        //统计总单词数量
        //这个排序是干什么用的？构建哈夫曼树？
        void SortVocab()
        {
            foreach (var word in vocab.Keys)
            {
                if (vocab[word].cn < min_count)
                {
                    vocab_word temp;
                    if (!vocab.TryRemove(word, out temp))
                    { Console.WriteLine("移除出现词数少的词失败"); }
                }
            }
            train_words = 0;
            foreach (var word in vocab.Values)
            {
                train_words += word.cn;
            }
            throw new Exception("还没有排序，onehotid还没有实现更新");
        }

        //从词表中删除出现次数小于min_reduce的词，没执行一次该函数min_reduce自动加一
        //这个是用来控制词表规模的，词表查过最大限度了，就把等于min_reduce的词删掉，
        //如果不行就继续把min_reduce+1的词删掉，再不行就min_reduce+2的词删掉，如此反复
        void ReduceVocab()
        {
            throw new Exception("方法未实现");

        }



        //从训练文件中获取所有词汇并构建词表和hash词表
        void LearnVocabFromTrainFile(string filename)
        {
            StreamReader file = new StreamReader(filename);
            while (!file.EndOfStream)
            {
                string[] fields = file.ReadLine().Split(' ');
                foreach (var word1 in fields)
                {
                    AddWordToVocab(word1);
                }
            }
            file.Close();

            //对词表进行排序，剔除词频低于阈值min_count的值，输出当前词表大小和总词数
            SortVocab();
            if (debug_mode > 0)
            {
                Console.WriteLine("Vocab size: %lld\n", vocab_size());
                Console.WriteLine("Words in train file: %lld\n", train_words);
            }

        }


        //将单词和对应的词频输出到文件中
        void SaveVocab(string filename)
        {
            StreamWriter file = new StreamWriter(filename);
            foreach (var item in vocab)
            {
                file.WriteLine(item.Value.word + "\t" + item.Value.cn);
            }
            file.Close();
        }



        //从词汇表文件中读词并构建词表和hash表
        //由于词汇表中的词语不存在重复，因此与LearnVocabFromTrainFile相比没有做重复词汇的检测
        void ReadVocab(string filename)
        {
            vocab.Clear();
            StreamReader file = new StreamReader(filename);
            while (!file.EndOfStream)
            {
                string[] fields = file.ReadLine().Split('\t');
                vocab_word word1 = new vocab_word();
                word1.word = fields[0];
                word1.cn = Convert.ToInt64(fields[1]);
                if (!vocab.TryAdd(fields[0], word1)) { Console.WriteLine("添加单词失败"); }
            }
        }

        //初始化神经网络结构
        void InitNet()
        {
            Int64 a, b;
            Int64 next_random = 1;
            //syn0存储的是词表中每个词的词向量
            //这里为syn0分配内存空间
            //调用posiz_memalign来获取一块数量为vocab_size * layer1_size，128byte页对齐的内存
            //其中layer1_size是词向量的长度
            syn0 = new double[vocab_size() * layer1_size];


            //多层Softmax回归
            if (hs == 1)
            {
                //syn1存储的是Haffman树中每个非叶节点的向量
                //这里为syn1分配内存空间               
                //初始化syn1为0
                throw new Exception("方法未实现");
            }

            //如果要使用负采样，则需要为syn1neg分配内存空间
            //syn1neg是负采样时每个词的辅助向量
            if (negative > 0)
            {
                //a = posix_memalign((void**)&syn1neg, 128, (Int64)vocab_size* layer1_size *sizeof(real));
                //if (syn1neg == NULL) { Console.WriteLine("Memory allocation failed\n"); exit(1); }
                ////初始化syn1neg为0
                //for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
                //        syn1neg[a * layer1_size + b] = 0;
                syn1neg = new double[vocab_size() * layer1_size];
            }

            for (a = 0; a < vocab_size(); a++)
            {
                for (b = 0; b < layer1_size; b++)
                {
                    next_random = next_random * (Int64)25214903917 + 11;
                    //初始化词向量syn0，每一维的值为[-0.5, 0.5]/layer1_size范围内的随机数
                    syn0[a * b] = (((next_random & 0xFFFF) / 65536.0) - 0.5) / layer1_size;
                }
            }
            //创建Haffman二叉树
            CreateBinaryTree();
        }

        //利用统计到的词频构建Haffman二叉树
        //根据Haffman树的特性，出现频率越高的词其二叉树上的路径越短，即二进制编码越短
        void CreateBinaryTree()
        {
            // throw new Exception("方法未实现");
            Console.WriteLine("CreateBinaryTree方法没有实现");
        }

        //该函数为线程函数，是训练算法代码实现的主要部分
        //默认在执行该线程函数前，已经完成词表排序、Haffman树的生成以及每个词的Haffman编码计算
        void TrainModelThread(Int64 id, string train_file)
        {
            //Int64 a, b, d;
            //Int64 cw;//cw：窗口长度（中心词除外）
            //word: 在提取句子时用来表示当前词在词表中的索引
            //last_word: 用于在窗口扫描辅助，记录当前扫描到的上下文单词
            //setence_length: 当前处理的句子长度
            //setence_position: 当前处理的单词在当前句子中的位置

            string word, last_word;
            Int64 sentence_length = 0, sentence_position = 0;
            //word_count: 当前线程当前时刻已训练的语料的长度
            //last_word_count: 当前线程上一次记录时已训练的语料长度
            Int64 word_count = 0, last_word_count = 0;
            //sen：当前从文件中读取的待处理句子，存放的是每个词
            string[] sen = new string[MAX_SENTENCE_LENGTH + 1];
            //l1：在skip-gram模型中，在syn0中定位当前词词向量的起始位置
            //l2：在syn1或syn1neg中定位中间节点向量或负采样向量的起始位置
            //target：在负采样中存储当前样本
            //label：在负采样中存储当前样本的标记
            Int64 l1, l2, c, local_iter = iter;
            string target;
            int label;
            //next_random：用来辅助生成随机数
            Random next_random = new Random();
            double f, g;

            //neu1：输入词向量，在CBOW模型中是Context(x)中各个词的向量和，在skip-gram模型中是中心词的词向量
            double[] neu1 = new double[layer1_size];
            //neuele：累计误差项
            double[] neu1e = new double[layer1_size];



            //这里是单进程，所以直接打开文件就行了
            StreamReader file = new StreamReader(train_file);

            //开始主循环
            while (!file.EndOfStream)
            {
                //每训练约10000词输出一次训练进度
                if (word_count - last_word_count > 10000)
                {
                    //word_count_actual是所有线程总共当前处理的词数
                    word_count_actual += word_count - last_word_count;
                    last_word_count = word_count;
                    //在初始学习率的基础上，随着实际训练词数的上升，逐步降低当前学习率（自适应调整学习率）
                    alpha = starting_alpha * (1 - word_count_actual / (double)(iter * train_words + 1));
                    //调整的过程中保证学习率不低于starting_alpha * 0.0001
                    if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
                }

                //从训练样本中取出一个句子，句子间以回车分割                
                if (sentence_length == 0)
                {
                    string[] words = file.ReadLine().Split(' ');
                    foreach (var one_word in words)
                    {
                        //对高频词进行随机下采样，丢弃掉一些高频词，能够使低频词向量更加准确，同时加快训练速度
                        //可以看作是一种平滑方法
                        if (sample > 0)
                        {
                            double ran = (Math.Sqrt(vocab[one_word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[one_word].cn;

                            //以1-ran的概率舍弃高频词
                            if (ran < next_random.NextDouble()) continue;
                        }
                        sen[sentence_length] = one_word;
                        sentence_length++;
                        //如果句子长度超出最大长度则截断
                        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                    }
                    //定位到句子头
                    sentence_position = 0;
                }



                //取出当前单词
                word = sen[sentence_position];
                if (!vocab.ContainsKey(word)) continue;

                //初始化输入词向量
                for (c = 0; c < layer1_size; c++) neu1[c] = 0;
                //初始化累计误差项
                for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

                //生成一个[0, window-1]的随机数，用来确定|context(w)|窗口的实际宽度（提高训练速率？）
                int ture_window_size = next_random.Next() % window;


                /********如果使用的是CBOW模型：输入是某单词周围窗口单词的词向量，来预测该中心单词本身*********/
                //              if (cbow)
                //              {
                //                  cw = 0;
                //                  //一个词的窗口为[setence_position - window + b, sentence_position + window - b]
                //                  //因此窗口总长度为 2*window - 2*b + 1
                //                  for (a = b; a < window * 2 + 1 - b; a++)
                //                      if (a != window)
                //                      {//去除窗口的中心词，这是我们要预测的内容，仅仅提取上下文
                //                          c = sentence_position - window + a;
                //                          if (c < 0) continue;
                //                          if (c >= sentence_length) continue;
                //                          //sen数组中存放的是句子中的每个词在词表中的索引
                //                          last_word = sen[c];
                //                          if (last_word == -1) continue;
                //                          //计算窗口中词向量的和
                //                          for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
                //                          //统计实际窗口中的有效词数
                //                          cw++;
                //                      }

                //                  if (cw)
                //                  {
                //                      //求平均向量和
                //                      for (c = 0; c < layer1_size; c++)
                //                      {
                //                          neu1[c] /= cw;
                //                      }


                //                      //如果采用分层softmax优化
                //                      //根据Haffman树上从根节点到当前词的叶节点的路径，遍历所有经过的中间节点
                //                      if (hs) for (d = 0; d < vocab[word].codelen; d++)
                //                          {
                //                              f = 0;
                //                              //l2为当前遍历到的中间节点的向量在syn1中的起始位置
                //                              l2 = vocab[word].point[d] * layer1_size;

                //                              //f为输入向量neu1与中间结点向量的内积
                //                              for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];

                //                              //检测f有没有超出Sigmoid函数表的范围
                //                              if (f <= -MAX_EXP) continue;
                //                              else if (f >= MAX_EXP) continue;
                //                              //如果没有超出范围则对f进行Sigmoid变换
                //                              else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

                //                              //g是梯度和学习率的乘积
                //                              //学习率越大，则错误分类的惩罚也越大，对中间向量的修正量也越大
                //                              //注意！word2vec中将Haffman编码为1的节点定义为负类，而将编码为0的节点定义为正类
                //                              //即一个节点的label = 1 - d
                //                              g = (1 - vocab[word].code[d] - f) * alpha;
                //                              //根据计算得到的修正量g和中间节点的向量更新累计误差
                //                              for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
                //                              //根据计算得到的修正量g和输入向量更新中间节点的向量值
                //                              //很好理解，假设vocab[word].code[d]编码为1，即负类，其节点label为1-1=0
                //                              //sigmoid函数得到的值为(0,1)范围内的数，大于label，很自然的，我们需要把这个中间节点的向量调小
                //                              //而此时的g = (label - f)*alpha是一个负值，作用在中间节点向量上时，刚好起到调小效果
                //                              //调小的幅度与sigmoid函数的计算值偏离label的幅度成正比
                //                              for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
                //                          }


                //                      //如果采用负采样优化
                //                      //遍历所有正负样本（1个正样本+negative个负样本）
                //                      if (negative > 0) for (d = 0; d < negative + 1; d++)
                //                          {
                //                              if (d == 0)
                //                              {
                //                                  //第一次循环处理的是目标单词，即正样本
                //                                  target = word;
                //                                  label = 1;
                //                              }
                //                              else
                //                              {
                //                                  //从能量表中随机抽取负样本
                //                                  next_random = next_random * (unsigned long long)25214903917 + 11;
                //                      target = table[(next_random >> 16) % table_size];
                //                      if (target == 0) target = next_random % (vocab_size - 1) + 1;
                //                      if (target == word) continue;
                //                      label = 0;
                //                  }
                //                  //在负采样优化中，每个词在syn1neg数组中对应一个辅助向量
                //                  //此时的l2为syn1neg中目标单词向量的起始位置
                //                  l2 = target * layer1_size;
                //                  f = 0;
                //                  //f为输入向量neu1与辅助向量的内积
                //                  for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
                //                  if (f > MAX_EXP) g = (label - 1) * alpha;
                //                  else if (f < -MAX_EXP) g = (label - 0) * alpha;
                //                  //g = (label - f)*alpha
                //                  else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                //                  //用辅助向量和g更新累计误差
                //                  for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                //                  //用输入向量和g更新辅助向量
                //                  for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
                //              }

                //              //根据获得的的累计误差，更新context(w)中每个词的词向量
                //              for (a = b; a < window * 2 + 1 - b; a++) if (a != window)
                //                  {
                //                      c = sentence_position - window + a;
                //                      if (c < 0) continue;
                //                      if (c >= sentence_length) continue;
                //                      last_word = sen[c];
                //                      if (last_word == -1) continue;
                //                      for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
                //                  }
                //          }
                //      }


                ///********如果使用的是skip-gram模型：输入是中心单词，来预测该单词的上下文*********/
                //else {

                //因为需要预测Context(w)中的每个词，因此需要循环2window - 2b + 1次遍历整个窗口
                //遍历时跳过中心单词
                for (int a = ture_window_size; a < window * 2 + 1 - ture_window_size; a++) if (a != window)
                    {
                        c = sentence_position - window + a;
                        if (c < 0) continue;
                        if (c >= sentence_length) continue;
                        //last_word为当前待预测的上下文单词
                        last_word = sen[c];
                        if (last_word == "") continue;
                        //l1为当前单词的词向量在syn0中的起始位置
                        l1 = vocab[last_word].oneHot * layer1_size;
                        //初始化累计误差
                        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;


                        //如果采用负采样优化
                        //遍历所有正负样本（1个正样本+negative个负样本）
                        //算法流程基本和CBOW的ns一样，也采用的是模型对称
                        if (negative > 0)
                            for (int d = 0; d < negative + 1; d++)
                            {
                                if (d == 0)
                                {
                                    target = word;
                                    label = 1;
                                }
                                else
                                {

                                    target = table[next_random.Next() % table_size()];//???????
                                    if (target == "") target = vocab[(table[next_random.Next() % (vocab_size() - 1) + 1])].word;
                                    if (target == word) continue;
                                    label = 0;
                                }
                                l2 = vocab[target].oneHot;
                                f = 0;
                                for (c = 0; c < layer1_size; c++)
                                {
                                    f += syn0[c + l1] * syn1neg[c + l2];
                                    ///？？？？？？？？？？

                                }
                                if (f > MAX_EXP) g = (label - 1) * alpha;
                                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                                for (c = 0; c < layer1_size; c++)
                                { neu1e[c] += g * syn1neg[c + l2]; }
                                for (c = 0; c < layer1_size; c++)
                                { syn1neg[c + l2] += g * syn0[c + l1]; }
                            }
                        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
                    }


                //完成了一个词的训练，句子中位置往后移一个词
                sentence_position++;
                //处理完一句句子后，将句子长度置为零，进入循环，重新读取句子并进行逐词计算
                if (sentence_position >= sentence_length)
                {
                    sentence_length = 0;
                    continue;
                }
            }

            file.Close();
        }
    }

}
