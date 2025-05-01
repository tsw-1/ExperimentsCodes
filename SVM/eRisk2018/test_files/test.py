from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    "positive61 positive61 surprise55 trust121",
    "positive1 positive62 surprise52 trust121",
]
 
tfidf_vec = TfidfVectorizer()
# 利用fit_transform得到TF-IDF矩阵
tfidf_matrix = tfidf_vec.fit_transform(corpus)
 
# 利用get_feature_names得到不重复的单词
print(tfidf_vec.get_feature_names())
print("=======================")
# 得到每个单词所对应的ID
print(tfidf_vec.vocabulary_)
print("==========================") 
# 输出TF-IDF矩阵
print(tfidf_matrix)
print("=========================================")
print(tfidf_matrix.toarray())

'''
[[0.         0.         0.         0.         0.24557576 0.
  0.47059455 0.         0.         0.47059455 0.         0.37102215
  0.         0.47059455 0.37102215 0.        ]
 [0.         0.         0.50676543 0.50676543 0.26445122 0.
  0.         0.         0.         0.         0.         0.
  0.50676543 0.         0.39953968 0.        ]
 [0.         0.         0.         0.         0.25246826 0.
  0.         0.48380259 0.48380259 0.         0.48380259 0.
  0.         0.         0.         0.48380259]
 [0.50676543 0.50676543 0.         0.         0.26445122 0.50676543
  0.         0.         0.         0.         0.         0.39953968
  0.         0.         0.         0.        ]]
'''

'''
train_subject64 - positive1: 0.40993714596036396
train_subject64 - positive61: 0.5761523551647353
train_subject64 - surprise55: 0.5761523551647353
train_subject64 - trust121: 0.40993714596036396
train_subject65 - positive1: 0.40993714596036396
train_subject65 - positive62: 0.5761523551647353
train_subject65 - surprise52: 0.5761523551647353
train_subject65 - trust121: 0.40993714596036396

train_subject64 - words sorted by tf-idf:
positive61: 0.5761523551647353
surprise55: 0.5761523551647353
positive1: 0.40993714596036396
trust121: 0.40993714596036396
train_subject65 - words sorted by tf-idf:
positive62: 0.5761523551647353
surprise52: 0.5761523551647353
positive1: 0.40993714596036396
trust121: 0.40993714596036396
'''
