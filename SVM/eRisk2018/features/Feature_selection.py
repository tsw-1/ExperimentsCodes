from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import scipy.sparse as sp
from time import time
import numpy as np
import json

def normalize(data):
    return data

def ngrams(train, test, labels, ntrain, mn=1, mx=1, nm=500, binary = False, donorm = False, stopwords = False, verbose = True, analyzer_char = False):
    
    ftrain = train
    ftest  = test
    y_train = labels
    
    t0 = time()
    analyzer_type = 'word'
    if analyzer_char:
        analyzer_type = 'char'
        
    if binary:
        vectorizer = CountVectorizer(ngram_range=(mn, mx))
    elif stopwords:
        vectorizer = TfidfVectorizer(max_n=mx,min_n=mn,stop_words='english',analyzer=analyzer_type,sublinear_tf=True)
    else:
        # TfidfVectorizer 是一个用于将文本数据转换为 TF-IDF 特征表示的类，将文本中的词语转换为向量形式，用于机器学习算法的输入
        vectorizer = TfidfVectorizer(ngram_range=(mn,mx),sublinear_tf=True,analyzer=analyzer_type)

    # 打印详细信息
    if verbose:
        print("extracting ngrams... where n is [%d,%d]" % (mn,mx))
    
    X_train = vectorizer.fit_transform(ftrain)
    X_test = vectorizer.transform(ftest)
    
    if verbose:
        print("done in %fs" % (time() - t0), X_train.shape, X_test.shape)

    y = np.array(y_train)    
    
    numFts = nm
    if numFts < X_train.shape[1]:
        ch2 = SelectKBest(chi2, k=numFts)
        X_train = ch2.fit_transform(X_train, y)
        X_test = ch2.transform(X_test)
        # 断言X_train是否为稀疏矩阵，若否，则程序中断执行
        assert sp.issparse(X_train) 
        
    names = vectorizer.get_feature_names_out()
    if verbose:
        print("Extracting best features by a chi-squared test.. ", X_train.shape, X_test.shape) 
    data={}
    data['train']=X_train.toarray().tolist()
    data['y']=y.tolist()
    data['X_test']=X_test.toarray().tolist()
    data['names']=names.tolist()   
    
    with open('/home/E22301339/depressionDetection/processedData/data2.json','w')as f:
        json.dump(data,f)
    print("........................end")
    return X_train, y, X_test

if __name__ == '__main__':
    print("..........................loading")

        #  读取训练集数据
    with open('/home/E22301339/depressionDetection/codes-author/data/train2.json', 'r') as f:
        data = json.load(f)
    data=list(data.values())

    for i in range(len(data)):
        data[i]=' '.join(data[i]).strip()
    train=data

    #  读取测试集数据
    with open('/home/E22301339/depressionDetection/codes-author/data/test2.json', 'r') as f:
        data = json.load(f)
    data=list(data.values())
    for i in range(len(data)):
        data[i]=' '.join(data[i]).strip()
    test=data

    #   训练集标签
    with open ('/home/E22301339/depressionDetection/features/risk_golden_truth.txt','r+')as f:
            list_data=[]
            list_total=[]  # 存储positive抑郁的用户id
            for item in f.readlines():
                list_data.append(item.strip())
    str=list_data[0][:-1].strip()
        # print(str)
        # print(list_data[0][len(list_data[0])-1])

    for item in list_data:
        if(item[len(item)-1]=='1'):
            list_total.append(item[:-1].strip())
        else:
            break
    with open('/home/E22301339/depressionDetection/processedData/user_features_train.json','r')as f:
            users=list(json.load(f).keys())
    labels=[]
    for user in users:
        if user in list_total:
            labels.append(1)
        else:
            labels.append(0)



    ntrain = len(train)

    verbose = True

    t0 = time()

    X_train, y_train, X_test = ngrams(train, test, labels, ntrain, 1, 2, 1600, donorm = True, verbose = verbose) #min_ngram, max_ngram, size of final vector
    X_train2, y_train, X_test2 = ngrams(train, test, labels, ntrain, 2, 2, 500, donorm = True, verbose = verbose) #char ngrams   

    TR_DDR = sp.hstack([X_train])
    TE_DDR = sp.hstack([X_test])

    TR_DDR = sp.hstack([X_train,X_train2]) #combine different vectors
    TE_DDR = sp.hstack([X_test,X_test2])

    data={}
    data['train']=TR_DDR.toarray().tolist()
    data['y']=labels
    data['X_test']=TE_DDR.toarray().tolist()  
    with open('/home/E22301339/depressionDetection/processedData/data2.json','w')as f:
        json.dump(data,f)

    if verbose:
        print("######## Total time for feature extraction: %fs" % (time() - t0), TR_DDR.shape, TE_DDR.shape)