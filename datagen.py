import numpy as np

def generate_batch_data_train(news_title,news_senti_cate,train_candidate,train_user_his,train_label,batch_size):
    id_list = np.arange(len(train_label))
    np.random.shuffle(id_list)
    y=train_label
    batches = [id_list[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]
    while (True):
        for i in batches:
            item = news_title[train_candidate[i]]
            user = news_title[train_user_his[i]] 
            item_senti = np.expand_dims(np.argmax(news_senti_cate[train_candidate[i]],axis=2),axis=2)
            user_senti = np.expand_dims(np.argmax(news_senti_cate[train_user_his[i]],axis=2),axis=2)
            user_mask = np.array(np.array(train_user_his[i],dtype='bool'),dtype='int32')
            yield ([item,user,user_senti,item_senti], [y[i],np.zeros_like(y[i])[:,:1],user_senti,item_senti],[np.ones_like(y[i])[:,0],np.ones_like(y[i])[:,0],user_mask,np.ones_like(item_senti)[:,:,0]])

def generate_batch_data_user(news_title, test_user_his, batch_size):
    idlist = np.arange(len(test_user_his))  
    batches = [idlist[range(batch_size*i, min(len(test_user_his), batch_size*(i+1)))] for i in range(len(test_user_his)//batch_size+1)]
    while (True):
        for i in batches: 
            yield ([news_title[test_user_his[i]]])
            