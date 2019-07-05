from  sklearn.feature_extraction.text import  TfidfVectorizer
import nltk
import jieba
tfidf_vec=TfidfVectorizer()
documents=[
    'this is the bayes document',
    'this is the second second document',
    'and the third one',
    'is this the document'
]
tfidf_matrix=tfidf_vec.fit_transform(documents)
#输出文档中所有不重复的词
print('不重复的词:',tfidf_vec.get_feature_names())
print('每个单词的ID：',tfidf_vec.vocabulary_)
print('每个单词的tfidf值：',tfidf_matrix.toarray())

#英文对文档进行分词
word_list=nltk.word_tokenize(text)
#标注单词词性
nltk.pos_tag(word_list)
#中文文档分词
word_list2=jieba.cut(text)

#加载停用词表
stop_words=[line.strip().decode('utf-8') for line in io.open('chineseStopWords.txt').readlines()]