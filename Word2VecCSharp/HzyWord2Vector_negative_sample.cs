using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Word2VecCSharp
{
    class Word_Dictionary
    {
        public static Dictionary<string, Int64> word_count;//每个单词出现的次数;
        public static Dictionary<string, int> word_oneHot;//每个单词的onehot;    
        public static List<string> table; //table 存储的是词频分布表
                                          
        //计算每个函数的词频分布表，在负采样中用到
        //负采样算法：带权采样思想。每个词的权重为l(w) = [counter(w)]^(3/4) / sum([counter(u)]^(3/4))，u属于词典D
        //每个词对应一个线段, 将[0,1]等距离划分成10^8，每次生成一个随机整数r，Table[r]就是一个样本。
        public static void InitUnigramTable()
        {

            double train_words_pow = 0;
            double d1;
            double power = 0.75;

            //为词频分布表分配内存空间，
            table = new List<string>();

            //遍历词表，根据词频计算累计值的分母
            foreach (var word in word_count)
            {
                train_words_pow += Math.Pow(word.Value, power); //  求和( 词频^0.75)
            }

            foreach (var word in word_count)
            {
                d1 = Math.Pow(word.Value, power) / train_words_pow; //  求和( 词频^0.75)
                for (double i = 0; i < d1; i = i + 1.0 / 1e8)
                {
                    table.Add(word.Key);
                }
            }
        }
    }
        
    class HzyWord2Vector_negative_sample
    {
        public static double[] syn0, syn1, syn1neg;
        public static double[] expTable;
        public static int negative = 5;
        public static int embedding_length = 100;


    }
}
