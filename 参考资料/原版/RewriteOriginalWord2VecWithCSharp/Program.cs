using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RewriteOriginalWord2VecWithCSharp
{
    //每个词的基本数据结构
    struct vocab_word
    {
        public Int64 cn;//词频，从训练集中计数得到或直接提供词频文件
        public List<int> point;//Haffman树中从根节点到该词的路径，存放的是路径上每个节点的索引
        public string word;//word为该词的字面值 
        public string code;//code为该词的haffman编码
        public string chodelen;//codelen为该词haffman编码的长度
    }
    class Hzyword2Vector
    {
        int MAX_STRING = 100;
        int EXP_TABLE_SIZE = 1000;
        int MAX_EXP = 6;
        int MAX_SENTENCE_LENGTH = 1000;
        int MAX_CODE_LENGTH = 40;

     public static SortedDictionary<string, vocab_word> vocab = new SortedDictionary<string, vocab_word>(new VocabCompare());
        

        int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

        //vocab_size为训练集中不同单词的个数，即词表的大小
        //layer1_size为词向量的长度
        Int64 vocab_size = 0, layer1_size = 100;

        Int64 train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;

        double alpha = 0.025, starting_alpha, sample = 1e-3;

        //syn0存储的是词表中每个词的词向量
        //syn1存储的是Haffman树中每个非叶节点的向量
        //syn1neg是负采样时每个词的辅助向量
        //expTable是提前计算好的Sigmond函数表
        double[] syn0, syn1, syn1neg, expTable;
        int hs = 0, negative = 5;


        List<string> table; //table 存储的是词频分布表
        static void Main(string[] args)
        {
        }


        //计算每个函数的词频分布表，在负采样中用到
        //负采样算法：带权采样思想。每个词的权重为l(w) = [counter(w)]^(3/4) / sum([counter(u)]^(3/4))，u属于词典D
        //每个词对应一个线段, 将[0,1]等距离划分成10^8，每次生成一个随机整数r，Table[r]就是一个样本。
        void InitUnigramTable()
        {
            int a;
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

        //从文件中读入一个词到word，以space' '，tab'\t'，EOL'\n'为词的分界符
        //截去一个词中长度超过MAX_STRING的部分
        //每一行的末尾输出一个</s>
        string ReadWord(StreamReader fin)
        {
            return "";
        }


        //返回一个词的hash值，由词的字面值计算得到，可能存在不同词拥有相同hash值的冲突情况
        int GetWordHash(string word)
        {
            return 0;
        }

        //返回一个词在词表中的位置，若不存在则返回-1
        //先计算词的hash值，然后在词hash表中，以该值为下标，查看对应的值
        //如果为-1说明这个词不存在索引，即不存在在词表中，返回-1
        //如果该索引在词表中对应的词与正在查找的词不符，说明发生了hash值冲突，按照开放地址法去寻找这个词
        int SearchVocab(string word)
        {
            return -1;
        }

        //从文件中读入一个词，并返回这个词在词表中的位置，相当于将之前的两个函数包装了起来
        int ReadWordIndex(StreamReader fin)
        {
            string word = ReadWord(fin);
            return SearchVocab(word);
        }

        //为一个词构建一个vocab_word结构对象，并添加到词表中
        //词频初始化为0，hash值用之前的函数计算，
        //返回该词在词表中的位置
        int AddWordToVocab(string word)
        {
            if (word.Length > MAX_STRING)
            {
                word = word.Substring(0, MAX_STRING);
            }
            if (!vocab.ContainsKey(word))
            {
                vocab_word one_vocab_word = new vocab_word();
                one_vocab_word.cn = 0;
                one_vocab_word.word = word;
                vocab.Add(word, one_vocab_word);
                vocab_size++;
            }
            return 0;
        }



        //输入两个词的结构体，返回二者的词频差值       
        public class VocabCompare : IComparer<vocab_word>
        {
            // Compares by Height, Length, and Width.
            public int Compare(vocab_word a, vocab_word b)
            {
                if (b.cn > a.cn) return 1;
                else if (b.cn < a.cn) return -1;
                else return 0;
            }
        }
        public class VocabCompare : IComparer<string>
        {
            // Compares by Height, Length, and Width.
            public int Compare(string x, string y)
            {
                vocab_word a = vocab[x];

                if (b.cn > a.cn) return 1;
                else if (b.cn < a.cn) return -1;
                else return 0;
            }
        }



        //统计词频，按照词频对词表中的项从大到小排序
        //这个排序是干什么用的？我知道了
        void SortVocab()
        {
            foreach (var word in vocab.Keys)
            {
                if(vocab[word].cn< min_count)
                {
                    vocab.Remove(word);
                }
            }
            vocab.sort()
        }
    }
}
