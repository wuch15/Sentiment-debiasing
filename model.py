import keras
from keras.layers import *
from keras.optimizers import *
from keras.models import Model
from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
              
        super(Attention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def senti_model(word_dict, emb_table,npratio=4):
    

    title_input = Input(shape=(30,), dtype='int32') 
    
    embedding_layer = Embedding(len(word_dict), 300, weights=[emb_table],trainable=True)
    emb_seq = embedding_layer(title_input)
    emb_seq=Dropout(0.2)(emb_seq)
    
    selfatt=Attention(20,20)([emb_seq,emb_seq,emb_seq])
    selfatt=Dropout(0.2)(selfatt)
    attention = Dense(200,activation='tanh')(selfatt)
    attention = Flatten()(Dense(1)(attention))
    attention_weight = Activation('softmax')(attention)   
    rep=Dot((1, 1))([selfatt, attention_weight])
                
    news_encoder = Model([title_input], rep) 
    
    senti_embedding_layer = Embedding(5, 256, trainable=True)
    senti_input = Input(shape=(1,), dtype='int32') 
    senti_rep=Dense(400,activation='tanh')(Flatten()(senti_embedding_layer(senti_input)))
    senti_encoder = Model([senti_input], senti_rep) 
    

    news_input = Input((50,30,))  
    news_senti_input = Input((50,1,))  
                                                       
    news_emb = TimeDistributed(news_encoder)(news_input) 
    news_emb_senti = TimeDistributed(senti_encoder)(news_senti_input) 
       
    news_hidden=Dropout(0.2)(Attention(20,20)([news_emb,news_emb,news_emb]))
    attention_n = Dense(200,activation='tanh')(news_hidden)
    attention_n = Flatten()(Dense(1)(attention_n))
    attention_n_weight = Activation('softmax')(attention_n)
    user_rep=Dot((1, 1))([news_hidden, attention_n_weight])
    user_model = Model([news_input], user_rep)
           
    news_senti_hidden=Dropout(0.2)(Attention(20,20)([news_emb_senti,news_emb_senti,news_emb_senti]))
    attention_n2 = Dense(200,activation='tanh')(news_senti_hidden)
    attention_n2 = Flatten()(Dense(1)(attention_n2))
    attention_n_weight2 = Activation('softmax')(attention_n2)
    user_rep_senti=Dot((1, 1))([news_senti_hidden, attention_n_weight2])
    
    #overalluser_rep=add([user_rep,user_rep_senti])
    
    candidates = keras.Input((1+npratio,30,)) 
    candidates_senti = keras.Input((1+npratio,1,)) 
    
    candidate_vecs = TimeDistributed(news_encoder)(candidates) 
    candidate_vecs_senti = TimeDistributed(senti_encoder)(candidates_senti) 
    #senti_cand = TimeDistributed(senti_pred)(candidate_vecs_senti)
    
    loss_orth_1=Lambda(lambda x:K.mean(K.sum(x[0]*x[1],axis=-1)/(1e-8+K.sqrt(K.sum(K.square(x[0]),axis=-1)*K.sum(K.square(x[1]),axis=-1))),axis=-1))([news_emb,news_emb_senti])
    loss_orth_2=Lambda(lambda x:K.mean(K.sum(x[0]*x[1],axis=-1)/(1e-8+K.sqrt(K.sum(K.square(x[0]),axis=-1)*K.sum(K.square(x[1]),axis=-1))),axis=-1))([candidate_vecs,candidate_vecs_senti])
    loss_orth_3=Lambda(lambda x:K.batch_dot(x[0],x[1],axes=-1)/(1e-8+K.sqrt(K.sum(K.square(x[0]),axis=1)*K.sum(K.square(x[1]),axis=1))))([user_rep,user_rep_senti])
    
    
    loss_orth=Lambda(lambda x:K.abs(x[0])+K.abs(x[1])+K.abs(x[2]))([loss_orth_1,loss_orth_2,loss_orth_3])
    
    def orthloss(y_true, y_pred):
        return K.mean(K.abs(loss_orth))
    
    logits_1 = dot([user_rep, candidate_vecs], axes=-1)
    logits_2 = dot([user_rep_senti, candidate_vecs_senti], axes=-1)
    logits= add([logits_1, logits_2])
    logits = Activation('softmax')(logits)    

    emb_input_1 = Input(shape=(50,400,), dtype='float32') 
    emb_input_2 = Input(shape=(5,400,), dtype='float32') 
    dense_1=Dense(400,activation='tanh')
    dense_2=Dense(5,activation='softmax')
    output1=TimeDistributed(dense_2)(TimeDistributed(dense_1)(emb_input_1))
    output2=TimeDistributed(dense_2)(TimeDistributed(dense_1)(emb_input_2))
    senti_pred = Model([emb_input_1,emb_input_2], [output1,output2])
    model_vec = Model([candidates,news_input ], [news_emb,candidate_vecs])
    
    gan_output_1,gan_output_2 = senti_pred(model_vec([candidates,news_input]))
    gan = Model([candidates,news_input,news_senti_input,candidates_senti], [logits,loss_orth,gan_output_1,gan_output_2])
    gan.compile(optimizer=Adam(lr=0.0001), metrics=['acc'],loss=['categorical_crossentropy',orthloss,'categorical_crossentropy','categorical_crossentropy'],loss_weights=[1.,1,-0.15,-0.15],sample_weight_mode=["None","None",'temporal','temporal'])
    senti_pred.compile(loss=['categorical_crossentropy','categorical_crossentropy'], optimizer=Adam(lr=0.0002), metrics=['acc'],sample_weight_mode=['temporal','temporal'])
    
    return model_vec, gan, senti_pred, user_model, news_encoder
    