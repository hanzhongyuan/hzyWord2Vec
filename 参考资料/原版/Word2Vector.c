//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

//ÿ���ʵĻ������ݽṹ
struct vocab_word {
  long long cn;		//��Ƶ����ѵ�����м����õ���ֱ���ṩ��Ƶ�ļ�
  int *point;		//Haffman���дӸ��ڵ㵽�ôʵ�·������ŵ���·����ÿ���ڵ������
  //wordΪ�ôʵ�����ֵ
  //codeΪ�ôʵ�haffman����
  //codelenΪ�ô�haffman����ĳ���
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
//�ʱ���������±��ʾ������ڴ˱��е�λ�ã�Ҳ��֮Ϊ������ڴʱ��е�����
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
//��hash����������±�Ϊÿ���ʵ�hashֵ���ɴʵ�����ֵASCII�����õ���vocab_hash[hash]�д洢���Ǹô��ڴʱ��е�����
int *vocab_hash;
//vocab_max_size��һ������������ÿ�ε��ʱ��С����vocab_max_sizeʱ��һ���Խ��ʱ��С����1000
//vocab_sizeΪѵ�����в�ͬ���ʵĸ��������ʱ�Ĵ�С
//layer1_sizeΪ�������ĳ���
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
//syn0�洢���Ǵʱ���ÿ���ʵĴ�����
//syn1�洢����Haffman����ÿ����Ҷ�ڵ������
//syn1neg�Ǹ�����ʱÿ���ʵĸ�������
//expTable����ǰ����õ�Sigmond������
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

//����ÿ�������������ֲ����ڸ��������õ�
void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  //Ϊ������table�����ڴ�ռ䣬����table_size�table_sizeΪһ���ȶ�����1e8
  table = (int *)malloc(table_size * sizeof(int));
  //�����ʱ����ݴ�Ƶ����������ֵ
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  //d1����ʾ�ѱ����ʵ�����ֵռ�������ı�
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  //a��������table������
  //i���ʱ������
  for (a = 0; a < table_size; a++) {
    //i�ŵ���ռ��table��aλ��
	table[a] = i;
	 //������ӳ����һ�����ʵ������ֲ�������õ��ʵ�����Խ����ռtable��λ�þ�Խ��
	//�����ǰ���ʵ������ܺ�d1С��ƽ��ֵ��i������ͬʱ����d1����֮��������ߵĻ�������i���䣬��ռ�ݸ����λ��
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
<span style="white-space:pre">	</span>//����ʱ������Ϻ�������û����������������ʣ�µ�λ���ôʱ������һ�������
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

//���ļ��ж���һ���ʵ�word����space' '��tab'\t'��EOL'\n'Ϊ�ʵķֽ��
//��ȥһ�����г��ȳ���MAX_STRING�Ĳ���
//ÿһ�е�ĩβ���һ��</s>
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

//����һ���ʵ�hashֵ���ɴʵ�����ֵ����õ������ܴ��ڲ�ͬ��ӵ����ͬhashֵ�ĳ�ͻ���
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

//����һ�����ڴʱ��е�λ�ã����������򷵻�-1
//�ȼ���ʵ�hashֵ��Ȼ���ڴ�hash���У��Ը�ֵΪ�±꣬�鿴��Ӧ��ֵ
//���Ϊ-1˵������ʲ��������������������ڴʱ��У�����-1
//����������ڴʱ��ж�Ӧ�Ĵ������ڲ��ҵĴʲ�����˵��������hashֵ��ͻ�����տ��ŵ�ַ��ȥѰ�������
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

//���ļ��ж���һ���ʣ�������������ڴʱ��е�λ�ã��൱�ڽ�֮ǰ������������װ������
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

//Ϊһ���ʹ���һ��vocab_word�ṹ���󣬲���ӵ��ʱ���
//��Ƶ��ʼ��Ϊ0��hashֵ��֮ǰ�ĺ������㣬
//���ظô��ڴʱ��е�λ��
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  //ÿ���ʱ���Ŀ�����������ֵʱ��һ����Ϊ���������һǧ���ʽṹ����ڴ�ռ�
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  //�����hashֵ�������ʲ�����ͻ����ʹ�ÿ��ŵ�ַ�������ͻ��Ϊ�����Ѱ��һ��hashֵ��λ��
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  //���ô��ڴʱ��е�λ�ø�������ҵ���hashֵ��λ
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

//���մ�Ƶ�Ӵ�С����
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

//ͳ�ƴ�Ƶ�����մ�Ƶ�Դʱ��е���Ӵ�С����
void SortVocab() {
  int a, size;
  unsigned int hash;
  //�Դʱ�������򣬽�</s>���ڵ�һ��λ��
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  //��ֵhash��
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    //�����ִ���С��min_count�ĴʴӴʱ���ȥ�������ִ�������min_count�����¼���hashֵ������hash�ʱ�
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
	//hashֵ����
      hash=GetWordHash(vocab[a].word);
	//hashֵ��ͻ���
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
	//�����ܴ���
      train_words += vocab[a].cn;
    }
  }
  //����ɾ���˴�Ƶ�ϵ͵Ĵʣ���������ʱ���ڴ�ռ�
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // ΪHaffman���Ĺ���Ԥ������ռ�
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

//�Ӵʱ���ɾ�����ִ���С��min_reduce�Ĵʣ�ûִ��һ�θú���min_reduce�Զ���һ
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  //����hash��
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  //����hash��
  for (a = 0; a < vocab_size; a++) {
    //hashֵ����
    hash = GetWordHash(vocab[a].word);
	//hashֵ��ͻ���
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

//����ͳ�Ƶ��Ĵ�Ƶ����Haffman������
//����Haffman�������ԣ�����Ƶ��Խ�ߵĴ���������ϵ�·��Խ�̣��������Ʊ���Խ��
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2;
  //�����ݴ�һ���ʵ����ڵ��Haffman��·��
  long long point[MAX_CODE_LENGTH];
  //�����ݴ�һ���ʵ�Haffman����
  char code[MAX_CODE_LENGTH];

  //�ڴ���䣬Haffman�������У�����n��Ҷ�ӽڵ㣬��һ������2n-1���ڵ�
  //count����ǰvocab_size��Ԫ��ΪHaffman����Ҷ�ӽڵ㣬��ʼ��Ϊ�ʱ������дʵĴ�Ƶ
  //count�����vocab_size��Ԫ��ΪHaffman���м������ɵķ�Ҷ�ӽڵ㣨�ϲ��ڵ㣩�Ĵ�Ƶ����ʼ��Ϊһ����ֵ1e15
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  //binary�����¼���ڵ�������丸�ڵ�Ķ����Ʊ��루0/1��
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  //paarent�����¼ÿ���ڵ�ĸ��ڵ�
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  //count����ĳ�ʼ��
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;

  //���²���Ϊ����Haffman�����㷨��Ĭ�ϴʱ��Ѿ�����Ƶ�ɸߵ�������
  //pos1��pos2Ϊ��Ϊ�ʱ��д�Ƶ�εͺ���͵������ʵ��±꣨��ʼʱ���Ǵʱ���ĩβ������
  //</s>��Ҳ����������
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  //������vocab_size-1��ѭ��������ÿ�����һ���ڵ㣬���ɹ�����������
  for (a = 0; a < vocab_size - 1; a++) {
    //�Ƚϵ�ǰ��pos1��pos2����min1i��min2i�м�¼��ǰ��Ƶ��С�ʹ�С�ڵ������
	//min1i��min2i������Ҷ�ӽڵ�Ҳ�����Ǻϲ�����м�ڵ�
    if (pos1 >= 0) {
	  //���count[pos1]�Ƚ�С����pos1���ƣ���֮pos2����
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
	  //���count[pos1]�Ƚ�С����pos1���ƣ���֮pos2����
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
	//��count����ĺ��δ洢�ϲ��ڵ�Ĵ�Ƶ������Сcount[min1i]�ʹ�Сcount[min2i]��Ƶ֮�ͣ�
    count[vocab_size + a] = count[min1i] + count[min2i];
	//��¼min1i��min2i�ڵ�ĸ��ڵ�
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    //������ÿ���ڵ�������ӽڵ��У���Ƶ�ϵ͵�Ϊ1�����Ƶ�ϸߵ�Ϊ0��
	binary[min2i] = 1;
  }

  //���ݵõ���Haffman������Ϊÿ���ʣ����е�Ҷ�ӽڵ㣩����Haffman����
  //����ҪΪ���дʷ�����룬���ѭ��vocab_size��
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
	  //��������Ѱ��Ҷ�ӽ��ĸ��ڵ㣬��binary�����д洢��·���Ķ����Ʊ������ӵ�code����ĩβ
      code[i] = binary[b];
	  //��point����������·���ڵ�ı��
      point[i] = b;
	  //Haffman����ĵ�ǰ���ȣ���Ҷ�ӽ�㵽��ǰ�ڵ�����
      i++;
      b = parent_node[b];
	  //����Haffman��һ����vocab_size*2-1���ڵ㣬����vocab_size*2-2Ϊ���ڵ�
      if (b == vocab_size * 2 - 2) break;
    }
	//�ڴʱ��и��¸ôʵ���Ϣ
	//Haffman����ĳ��ȣ���Ҷ�ӽ�㵽���ڵ�����
    vocab[a].codelen = i;
	//Haffman·���д洢���м�ڵ���Ҫ�����ڵõ��Ļ����ϼ�ȥvocab_size��������Ҷ�ӽ�㣬�������м�ڵ��еı��
	//�������ڸ��ڵ�ı��Ϊ(vocab_size*2-2) - vocab_size = vocab_size - 2
    vocab[a].point[0] = vocab_size - 2;
	//Haffman�����·����Ӧ���ǴӸ��ڵ㵽Ҷ�ӽ��ģ������Ҫ��֮ǰ�õ���code��point���з���
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

//��ѵ���ļ��л�ȡ���дʻ㲢�����ʱ��hash��
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;

  //��ʼ��hash�ʱ�
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

  //��ѵ���ļ�
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }

  //��ʼ���ʱ��С
  vocab_size = 0;
  //��</s>��ӵ��ʱ����ǰ��
  AddWordToVocab((char *)"</s>");

 //��ʼ����ѵ���ļ�
  while (1) {
	//���ļ��ж���һ����
    ReadWord(word, fin);
    if (feof(fin)) break;
	//���ܴ�����һ���������ǰѵ����Ϣ
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
	//����������ڴʱ��е�λ��
    i = SearchVocab(word);
    //����ʱ��в���������ʣ��򽫸ô���ӵ��ʱ��У���������hash���е�ֵ����ʼ����ƵΪ1����֮����Ƶ��һ
	if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
	//����ʱ��С�������ޣ�����һ�δʱ�ɾ������,����ǰ��Ƶ��͵Ĵ�ɾ��
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  //�Դʱ���������޳���Ƶ������ֵmin_count��ֵ�������ǰ�ʱ��С���ܴ���
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  //��ȡѵ���ļ��Ĵ�С���ر��ļ����
  file_size = ftell(fin);
  fclose(fin);
}

//�����ʺͶ�Ӧ�Ĵ�Ƶ������ļ���
void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

//�Ӵʻ���ļ��ж��ʲ������ʱ��hash��
//���ڴʻ���еĴ��ﲻ�����ظ��������LearnVocabFromTrainFile���û�����ظ��ʻ�ļ��
void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  //�򿪴ʻ���ļ�
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  //��ʼ��hash�ʱ�
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;

  //��ʼ����ʻ���ļ�
  while (1) {
	//���ļ��ж���һ����
    ReadWord(word, fin);
    if (feof(fin)) break;
	//���ô���ӵ��ʱ��У���������hash���е�ֵ����ͨ������Ĵʻ���ļ��е�ֵ����������ʵĴ�Ƶ
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  //�Դʱ���������޳���Ƶ������ֵmin_count��ֵ�������ǰ�ʱ��С���ܴ���
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  //��ѵ���ļ������ļ�ָ�������ļ�ĩβ����ȡѵ���ļ��Ĵ�С
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  //�ر��ļ����
  fclose(fin);
}

//��ʼ��������ṹ
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  //syn0�洢���Ǵʱ���ÿ���ʵĴ�����
  //����Ϊsyn0�����ڴ�ռ�
  //����posiz_memalign����ȡһ������Ϊvocab_size * layer1_size��128byteҳ������ڴ�
  //����layer1_size�Ǵ������ĳ���
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}

  //���Softmax�ع�
  if (hs) {
    //syn1�洢����Haffman����ÿ����Ҷ�ڵ������
    //����Ϊsyn1�����ڴ�ռ�
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    //��ʼ��syn1Ϊ0
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }

  //���Ҫʹ�ø�����������ҪΪsyn1neg�����ڴ�ռ�
  //syn1neg�Ǹ�����ʱÿ���ʵĸ�������
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    //��ʼ��syn1negΪ0
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    //��ʼ��������syn0��ÿһά��ֵΪ[-0.5, 0.5]/layer1_size��Χ�ڵ������
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  //����Haffman������
  CreateBinaryTree();
}


//�ú���Ϊ�̺߳�������ѵ���㷨����ʵ�ֵ���Ҫ����
//Ĭ����ִ�и��̺߳���ǰ���Ѿ���ɴʱ�����Haffman���������Լ�ÿ���ʵ�Haffman�������
void *TrainModelThread(void *id) {
  long long a, b, d;
  //cw�����ڳ��ȣ����Ĵʳ��⣩
  long long cw;
  //word: ����ȡ����ʱ������ʾ��ǰ���ڴʱ��е�����
  //last_word: �����ڴ���ɨ�踨������¼��ǰɨ�赽�������ĵ���
  //setence_length: ��ǰ����ľ��ӳ���
  //setence_position: ��ǰ����ĵ����ڵ�ǰ�����е�λ��
  long long word, last_word, sentence_length = 0, sentence_position = 0;
  //word_count: ��ǰ�̵߳�ǰʱ����ѵ�������ϵĳ���
  //last_word_count: ��ǰ�߳���һ�μ�¼ʱ��ѵ�������ϳ���
  long long word_count = 0, last_word_count = 0;
  //sen����ǰ���ļ��ж�ȡ�Ĵ�������ӣ���ŵ���ÿ�����ڴʱ��е�����
  long long sen[MAX_SENTENCE_LENGTH + 1];
  //l1����skip-gramģ���У���syn0�ж�λ��ǰ�ʴ���������ʼλ��
  //l2����syn1��syn1neg�ж�λ�м�ڵ������򸺲�����������ʼλ��
  //target���ڸ������д洢��ǰ����
  //label���ڸ������д洢��ǰ�����ı��
  long long l1, l2, c, target, label, local_iter = iter;
  //next_random�������������������
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  //neu1���������������CBOWģ������Context(x)�и����ʵ������ͣ���skip-gramģ���������ĴʵĴ�����
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  //neuele���ۼ������
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));

  FILE *fi = fopen(train_file, "rb");

  //ÿ�����̶�Ӧһ���ı������ݵ�ǰ�̵߳�id�ҵ����̶߳�Ӧ�ı��ĳ�ʼλ��
  //file_size����֮ǰLearnVocabFromTrainFile��ReadVocab�����л�ȡ��ѵ���ļ��Ĵ�С
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

  //��ʼ��ѭ��
  while (1) {
    //ÿѵ��Լ10000�����һ��ѵ������
    if (word_count - last_word_count > 10000) {
      //word_count_actual�������߳��ܹ���ǰ����Ĵ���
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        //�����Ϣ������
        //��ǰ��ѧϰ��alpha��
        //ѵ���ܽ��ȣ���ǰѵ�����ܴ���/(��������*ѵ�������ܴ���)+1����
        //ÿ���߳�ÿ�봦��Ĵ���
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      //�ڳ�ʼѧϰ�ʵĻ����ϣ�����ʵ��ѵ���������������𲽽��͵�ǰѧϰ�ʣ�����Ӧ����ѧϰ�ʣ�
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      //�����Ĺ����б�֤ѧϰ�ʲ�����starting_alpha * 0.0001
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }

    //��ѵ��������ȡ��һ�����ӣ����Ӽ��Իس��ָ�
    if (sentence_length == 0) {
      while (1) {
        //���ļ��ж���һ���ʣ����ô��ڴʱ��е���������word
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        //���������ʱ�س�����ʾ���ӽ���
        if (word == 0) break;
        //�Ը�Ƶ�ʽ�������²�����������һЩ��Ƶ�ʣ��ܹ�ʹ��Ƶ����������׼ȷ��ͬʱ�ӿ�ѵ���ٶ�
        //���Կ�����һ��ƽ������
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          //��1-ran�ĸ���������Ƶ��
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        //������ӳ��ȳ�����󳤶���ض�
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      //��λ������ͷ
      sentence_position = 0;
    }

    //�����ǰ�̴߳���Ĵ�����������Ӧ�ô�������ֵ����ô��ʼ��һ�ֵ���
    //����������������ޣ���ֹͣ����
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    //ȡ����ǰ����
    word = sen[sentence_position];
    if (word == -1) continue;
    //��ʼ�����������
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    //��ʼ���ۼ������
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    //����һ��[0, window-1]�������������ȷ��|context(w)|���ڵ�ʵ�ʿ�ȣ����ѵ�����ʣ���
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;


    /********���ʹ�õ���CBOWģ�ͣ�������ĳ������Χ���ڵ��ʵĴ���������Ԥ������ĵ��ʱ���*********/
    if (cbow) {
      cw = 0;
      //һ���ʵĴ���Ϊ[setence_position - window + b, sentence_position + window - b]
      //��˴����ܳ���Ϊ 2*window - 2*b + 1
      for (a = b; a < window * 2 + 1 - b; a++)
        if (a != window) {//ȥ�����ڵ����Ĵʣ���������ҪԤ������ݣ�������ȡ������
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          //sen�����д�ŵ��Ǿ����е�ÿ�����ڴʱ��е�����
          last_word = sen[c];
          if (last_word == -1) continue;
          //���㴰���д������ĺ�
          for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
          //ͳ��ʵ�ʴ����е���Ч����
          cw++;
        }

      if (cw) {
        //��ƽ��������
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;


        //������÷ֲ�softmax�Ż�
        //����Haffman���ϴӸ��ڵ㵽��ǰ�ʵ�Ҷ�ڵ��·�����������о������м�ڵ�
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          //l2Ϊ��ǰ���������м�ڵ��������syn1�е���ʼλ��
          l2 = vocab[word].point[d] * layer1_size;

          //fΪ��������neu1���м����������ڻ�
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];

          //���f��û�г���Sigmoid������ķ�Χ
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          //���û�г�����Χ���f����Sigmoid�任
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

          //g���ݶȺ�ѧϰ�ʵĳ˻�
          //ѧϰ��Խ����������ĳͷ�ҲԽ�󣬶��м�������������ҲԽ��
          //ע�⣡word2vec�н�Haffman����Ϊ1�Ľڵ㶨��Ϊ���࣬��������Ϊ0�Ľڵ㶨��Ϊ����
          //��һ���ڵ��label = 1 - d
          g = (1 - vocab[word].code[d] - f) * alpha;
          //���ݼ���õ���������g���м�ڵ�����������ۼ����
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          //���ݼ���õ���������g���������������м�ڵ������ֵ
          //�ܺ���⣬����vocab[word].code[d]����Ϊ1�������࣬��ڵ�labelΪ1-1=0
          //sigmoid�����õ���ֵΪ(0,1)��Χ�ڵ���������label������Ȼ�ģ�������Ҫ������м�ڵ��������С
          //����ʱ��g = (label - f)*alpha��һ����ֵ���������м�ڵ�������ʱ���պ��𵽵�СЧ��
          //��С�ķ�����sigmoid�����ļ���ֵƫ��label�ķ��ȳ�����
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }


        //������ø������Ż�
        //������������������1��������+negative����������
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            //��һ��ѭ���������Ŀ�굥�ʣ���������
            target = word;
            label = 1;
          } else {
            //���������������ȡ������
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          //�ڸ������Ż��У�ÿ������syn1neg�����ж�Ӧһ����������
          //��ʱ��l2Ϊsyn1neg��Ŀ�굥����������ʼλ��
          l2 = target * layer1_size;
          f = 0;
          //fΪ��������neu1�븨���������ڻ�
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          //g = (label - f)*alpha
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          //�ø���������g�����ۼ����
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          //������������g���¸�������
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
        }

        //���ݻ�õĵ��ۼ�������context(w)��ÿ���ʵĴ�����
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
        }
      }
    }


    /********���ʹ�õ���skip-gramģ�ͣ����������ĵ��ʣ���Ԥ��õ��ʵ�������*********/
    else {

      //��Ϊ��ҪԤ��Context(w)�е�ÿ���ʣ������Ҫѭ��2window - 2b + 1�α�����������
      //����ʱ�������ĵ���
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        //last_wordΪ��ǰ��Ԥ��������ĵ���
        last_word = sen[c];
        if (last_word == -1) continue;
        //l1Ϊ��ǰ���ʵĴ�������syn0�е���ʼλ��
        l1 = last_word * layer1_size;
        //��ʼ���ۼ����
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;


        //������÷ֲ�softmax�Ż�
        //����Haffman���ϴӸ��ڵ㵽��ǰ�ʵ�Ҷ�ڵ��·�����������о������м�ڵ�
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          //ע�⣡�����õ���ģ�ͶԳƣ�p(u|w) = p(w|u)������wΪ���Ĵʣ�uΪcontext(w)��ÿ����
          //Ҳ����skip-gram��Ȼ�Ǹ����Ĵ�Ԥ�������ģ�����ѵ����ʱ������������Ԥ�����Ĵ�
          //��CBOW��ͬ���������u�ǵ����ʵĴ������������Ǵ�������֮��
          //�㷨���̻�����CBOW��hsһ�������ﲻ��׸��
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (1 - vocab[word].code[d] - f) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }


        //������ø������Ż�
        //������������������1��������+negative����������
        //�㷨���̻�����CBOW��nsһ����Ҳ���õ���ģ�ͶԳ�
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }

    //�����һ���ʵ�ѵ����������λ��������һ����
    sentence_position++;
    //������һ����Ӻ󣬽����ӳ�����Ϊ�㣬����ѭ�������¶�ȡ���Ӳ�������ʼ���
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }

  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

//������ģ��ѵ�����̺���
void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  //�������̣߳��߳���Ϊnum_threads
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  //���ó�ʼѧϰ��
  starting_alpha = alpha;
  //����дʻ���ļ�������м������ɴʱ��hash�������ѵ���ļ��л��
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  //������Ҫ�����Խ��ʱ��еĴʺʹ�Ƶ������ļ�
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  //��ʼ��ѵ������
  InitNet();
  //���ʹ�ø������Ż�������Ҫ��ʼ��������
  if (negative > 0) InitUnigramTable();
  //��ʼ��ʱ
  start = clock();
  //����ѵ���߳�
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");

  //���classes����Ϊ0����������д��������ļ���
  if (classes == 0) {
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  }

  //���classes������Ϊ0������Ҫ�Դ���������K-means���࣬�������
  //classesΪ���Ҫ�ֳɵ���ĸ���
  else {
    //clcn��������
    //iter���ܵ�������
    //closeid�������洢�����������ĳ�������������
    int clcn = classes, iter = 10, closeid;
    //centcn������ÿ����ĵ�����
    int *centcn = (int *)malloc(classes * sizeof(int));
    //cl��ÿ����������������
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    //x�������洢ÿ�μ���õ��Ĵ������������ĵ��ڻ���ֵԽ��˵������Խ��
    //closev�����������ڻ������������
    real closev, x;
    //cent��ÿ�������������
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));

    //�ȸ����е������ָ����
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;

    //һ������iter��
    for (a = 0; a < iter; a++) {
      //��ʼ����������������Ϊ0
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      //��ʼ��ÿ���ຬ�еĵ�����Ϊ1
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      //���ղ���������������ͬһ����Ĵ�������ӣ����Ҽ�������ÿ����Ĵ���
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }

      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          //����ÿ�����ƽ����������
          cent[layer1_size * b + c] /= centcn[b];
          //closevΪ��ƽ�����������Ķ�������ƽ��
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        //��closev��������ʱ��closev��Ϊ��ƽ�����������Ķ�����
        closev = sqrt(closev);
        //�õõ��ķ����������������й�һ��
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }

      //�����ʱ��е�ÿ���ʣ�Ϊ�����·�������������
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          //�Դ������͹�һ�����������������ڻ�
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          //�ڻ�Խ��˵������֮�����Խ��
          //ȡ��������������ʵĴ������ڻ�����һ���࣬���ʷֵ��������
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }

    //������ε������𽥻Ὣ����������ȷ���࿿£
    //���K-means���������ļ���
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

//������ȱʧʱ�������ʾ��Ϣ
int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");

    //�����ļ����ѷִʵ�����
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");

    //����ļ������������ߴʾ���
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");

    //��������ά�ȣ�Ĭ��ֵ��100
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");

    //���ڴ�С��Ĭ����5
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");

    //�趨�ʳ���Ƶ�ʵ���ֵ�����ڳ����ֵĴʻᱻ����²���
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");

    //�Ƿ����softmax��ϵ
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");

    //��������������Ĭ����0��ͨ��ʹ��5-10��0��ʾ��ʹ�á�
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");

    //�������߳�����
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");

    //��С��ֵ�����ڳ��ִ������ڸ�ֵ�Ĵʣ��ᱻ��������
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");

    //ѧϰ���ʳ�ʼֵ��Ĭ����0.025
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");

    //�������𣬶����Ǵ�����
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");

    //debugģʽ��Ĭ����2����ʾ��ѵ�������л����������Ϣ
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");

    //�Ƿ���binaryģʽ�������ݣ�Ĭ����0����ʾ��
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");

    //����ʻ㵽����ļ�
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");

    //�ʻ�Ӹ��ļ���ȡ����������ѵ����������
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");

    //�Ƿ����continuous bag of words�㷨��Ĭ����0����ʾ������һ����skip-gram���㷨��
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");

    //����ʹ������
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++)
  {
	//expTable[i] = exp((i -500)/ 500 * 6) �� e^-6 ~ e^6
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    //expTable[i] = 1/(1+e^6) ~ 1/(1+e^-6)�� 0.01 ~ 1 ������
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
