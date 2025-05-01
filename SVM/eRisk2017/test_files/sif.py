import numpy as np
from sklearn.decomposition import TruncatedSVD
import fasttext
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_weighted_average(We, x, w):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, We.shape[1]))
    for i in xrange(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb

def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_

def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, w, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    if  params.rmpc > 0:
        emb = remove_pc(emb, params.rmpc)
    return emb

def get_word_vectors(word_list, word_vectors):
    word_indices = [word_vectors.index(word) for word in word_list]
    return np.array(word_vectors)[word_indices]

def get_sentence_index(word_list, word_index):
    return [word_index[word] for word in word_list]

def get_tfidf_weights(sentences):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names()
    weights = []
    for i in range(len(sentences)):
        sentence_weights = {}
        for j, word in enumerate(feature_names):
            tfidf_value = tfidf_matrix[i, j]
            if tfidf_value > 0:
                sentence_weights[word] = tfidf_value
        weights.append(sentence_weights)
    return weights

if __name__ == '__main__':
    # text="Hello this is a good day"
    # vectorizer = TfidfVectorizer(use_idf=False)  # 设置 use_idf=False，只计算 TF
    # train = vectorizer.fit_transform([text])
    # tf_values = train.toarray()[0].tolist()  # 将稀疏矩阵转换为列表
    # print(tf_values)




    # model = fasttext.load_model('/home/E22301339/fastText/fastText/cc.en.300.bin')
    # We=[]
    # for item in text:
    #     vector=model[item]
    #     We.append(vector)
    # We=np.array(We)
    
        # 示例数据
    word_vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5]
    ]
    sentence = 'This is a sample sentence.'
    sentences = [sentence]

    # 获取词向量
    word_list = sentence.split()
    word_vectors = get_word_vectors(word_list, word_vectors)
    print("Word Vectors:", word_vectors)

    # 构建句子索引
    word_index = {word: index for index, word in enumerate(word_list)}
    sentence_index = get_sentence_index(word_list, word_index)
    print("Sentence Index:", sentence_index)

    # 计算 TF-IDF 权重
    weights = get_tfidf_weights(sentences)
    print("TF-IDF Weights:", weights)




