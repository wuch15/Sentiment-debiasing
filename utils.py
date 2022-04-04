from numpy.linalg import cholesky
import numpy as np


def senti2cate(x):
    if x<=-0.6:
        return 0
    elif x>-0.6 and x<=-0.2:
        return 1
    elif x>-0.2 and x<0.2:
        return 2
    elif x>=0.2 and x<0.6:
        return 3
    elif x>=0.6:
        return 4            



def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def auc(label,score):
    label=np.array(label)
    score=np.array(score)
    false_score = score[label==0]
    positive_score = score[label==1]
    num_positive = (label==1).sum()
    num_negative = (label==0).sum()
    positive_score = positive_score.reshape((num_positive,1))
    positive_score = np.repeat(positive_score,num_negative,axis=1)
    false_score = false_score.reshape((1,num_negative))
    false_score = np.repeat(false_score,num_positive,axis=0)
    return 1-((positive_score<false_score).mean()+0.5*(positive_score==false_score).mean())

    
def embedding(embfile,word_dict):
    emb_dict = {}  
    with open(embfile,'rb')as f: 
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            data = line.split()
            word = data[0].decode() 
            if len(word) != 0:
                vec = [float(x) for x in data[1:]]
                if word in word_dict:
                    emb_dict[word] = vec 
    
    emb_table = [0]*len(word_dict)
    dummy = np.zeros(300,dtype='float32')
    
    all_emb = []
    for i in emb_dict:
        emb_table[word_dict[i][0]] = np.array(emb_dict[i],dtype='float32')
        all_emb.append(emb_table[word_dict[i][0]])
    all_emb = np.array(all_emb,dtype='float32')
    mu = np.mean(all_emb, axis=0)
    Sigma = np.cov(all_emb.T)  
    norm = np.random.multivariate_normal(mu, Sigma, 1)
    for i in range(len(emb_table)):
        if type(emb_table[i]) == int:
            emb_table[i] = np.reshape(norm, 300)
    emb_table[0] = np.random.uniform(-0.03,0.03,size=(300,))
    emb_table = np.array(emb_table,dtype='float32')  
    return emb_table