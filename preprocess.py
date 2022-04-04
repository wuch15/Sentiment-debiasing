import os
import random
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from keras.utils import to_categorical
from utils import *

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

def process_news(dirlist):
    news_data=[]
    for folder in dirlist:
        with open(os.path.join(folder,'news.tsv'),encoding='utf-8') as f:
            news_data+=f.readlines()
    news={}
    for line in news_data:
        linesplit=line.strip().split('\t')
        if linesplit[0] not in news:
            # category, subcategory, title, title sentiment
            news[linesplit[0]]=[linesplit[1],linesplit[2],word_tokenize(linesplit[3].lower()),sid.polarity_scores(linesplit[3])['compound']]
    
    news_index={'NULL':0}  
    word_dict={'PADDING':[0,999999]}
    news_title=[[0]*30] 
    news_senti=[[0.]]
    news_senti_cate=[[0,0,1,0,0]] 
    
    for news_id in news:
        news_index[news_id]=len(news_index)
        title=[]
        for word in news[news_id][2]:
            if word not in word_dict:
                word_dict[word]=[len(word_dict),1]
            else:
                word_dict[word][1]+=1 
            title.append(word_dict[word][0])
            
        title=title[:30]
        news_title.append(title+[0]*(30-len(title)))
        news_senti.append([news[news_id][3]])
        news_senti_cate.append(to_categorical(senti2cate(news[news_id][3]),5))
        
    news_title=np.array(news_title,dtype='int32')  
    news_senti=np.array(news_senti,dtype='float32') 
    news_senti_cate=np.array(news_senti_cate,dtype='int32') 

    return news_index, news_title, word_dict, news_senti, news_senti_cate


def new_sample(array,ratio):
    if ratio >len(array):
        return random.sample(array*(ratio//len(array)+1),ratio)
    else:
        return random.sample(array,ratio)
    

def process_users(root_dir,news_index,is_train,npratio=4):

    with open(os.path.join(root_dir,'behaviors_1.tsv'),encoding='utf-8') as f:
        user_data=f.readlines()
    candidate=[]    
    label=[]
    user_his=[]
    if is_train:
        for user in user_data:
            userline=user.replace('\n','').split('\t')
            clickids=[news_index[x] for x in userline[3].split()][-50:]
            pdoc=[news_index[x.split('-')[0]] for x in userline[4].split() if x.split('-')[1]=='1']
            ndoc=[news_index[x.split('-')[0]] for x in userline[4].split() if x.split('-')[1]=='0']
            
            for doc in pdoc:
                negd=new_sample(ndoc,npratio)
                negd.append(doc)
                candidate_label=[0]*npratio+[1]
                candidate_order=list(range(npratio+1))
                random.shuffle(candidate_order)
                candidate_shuffle=[]
                candidate_label_shuffle=[]
                for i in candidate_order:
                    candidate_shuffle.append(negd[i])
                    candidate_label_shuffle.append(candidate_label[i])
                candidate.append(candidate_shuffle)
                label.append(candidate_label_shuffle)
                user_his.append(clickids+[0]*(50-len(clickids))) 
        candidate=np.array(candidate,dtype='int32')
        label=np.array(label,dtype='int32')
        user_his=np.array(user_his,dtype='int32')
        return candidate,user_his,label
    else:
        test_index=[]
        for user in user_data:
            userline=user.replace('\n','').split('\t')
            clickids=[news_index[x] for x in userline[3].split()][-50:]
            doc=[news_index[x.split('-')[0]] for x in userline[4].split()]
            index=[]
            index.append(len(candidate))
            user_his.append(clickids+[0]*(50-len(clickids)))
            for doc in doc: 
                candidate.append(doc)
            index.append(len(candidate))
            for x in userline[4].split():
                label.append(int(x.split('-')[1]))
            
            test_index.append(index)
        candidate=np.array(candidate,dtype='int32')
        label=np.array(label,dtype='int32')
        user_his=np.array(user_his,dtype='int32')        
        return candidate,user_his,label,test_index
