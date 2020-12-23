# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:50:08 2018

@author: peternapolean
"""

import scipy.sparse as sp
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd
import os
import gzip
import pickle
from scipy.sparse import csr_matrix as sparse_matrix
from surprise.prediction_algorithms.matrix_factorization import SVD, SVDpp, NMF
from surprise import Dataset,Reader
from random import sample

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)

class DataLoader(object):
    def __init__(self,category,save_name):
        self.category=category
        self.max_user=10000 #maximum number of user
        self.price_dict={}
        self.price_dict_temp={}
        self.cate_dict={}
        self.cate_dict_temp={}
        self.top_value=15 # top x features in SVD
        self.model=NMF()
        self.topk=500 #maximum items in each category, finding the top k popular
        self.max_price={}
        self.save_path= os.path.join("..", "feature", save_name)
        if not os.path.isfile(self.save_path):
            self.load_data() #load raw data
            #self.create_user_item_matrix()
            self.create_ratings()
            self.gen_new_price_dict()
            self.save_data(self.save_path) #save the feature
        else:
            self.load(self.save_path) #load the feature
        
    
    def load_ratings(self, filename):
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        return ratings
    
    def load_prices(self,filename):
        price_dict = {}
        num_no_price=0
        for review in parse(os.path.join("..", "data", filename)):
            try:
                price=review['price']
                asin=review['asin']
                v=list(review['salesRank'].values())[0]
                if v<self.topk:
                    price_dict[asin]=price
            except:
                num_no_price+=1
                continue
        print("filename:",filename)
        print("length of price dict:", len(price_dict))
        print("# of items without price", num_no_price)
        return price_dict
    
    def load_data(self):
        print("Loading data:")
        for i in self.category:
            ratings_name= "ratings_"+i+".csv"
            price_name="meta_"+i+".json.gz"
            ratings_temp=self.load_ratings(ratings_name)
            print(len(ratings_temp))
            price_temp=self.load_prices(price_name)
            ratings_temp=ratings_temp[ratings_temp['item'].isin(price_temp.keys())]
            print(len(ratings_temp))
            self.price_dict_temp.update(price_temp)
            self.max_price[i]=max(list(price_temp.values()))
            cate_temp={}
            for j in price_temp.keys():
                cate_temp[j]=i
            self.cate_dict_temp.update(cate_temp)
            price_temp.clear()
            try:
                self.ratings=pd.merge(self.ratings,ratings_temp, how='outer')
            except:
                self.ratings=ratings_temp
        print(self.max_price)
        
    #old method
    def create_user_item_matrix(self, user_key="user",item_key="item"):
        n = len(set(self.ratings[user_key]))
        d = len(set(self.ratings[item_key]))
        self.user_mapper = dict(zip(np.unique(self.ratings[user_key]), list(range(n))))
        self.item_mapper = dict(zip(np.unique(self.ratings[item_key]), list(range(d))))

        self.user_inverse_mapper = dict(zip(list(range(n)), np.unique(self.ratings[user_key])))
        self.item_inverse_mapper = dict(zip(list(range(d)), np.unique(self.ratings[item_key])))

        self.user_ind = [self.user_mapper[i] for i in self.ratings[user_key]]
        self.item_ind = [self.item_mapper[i] for i in self.ratings[item_key]]

        self.ratings_matrix = sparse_matrix((self.ratings["rating"]-3, (self.user_ind, self.item_ind)), shape=(n,d))
        print("user-item matrix generated.")
    
    def create_ratings(self):
        #C=MBRecsys(self.ratings_matrix,top_value)
        S=set(self.ratings['user'])
        S=sample(S,self.max_user)
        n = len(S)
        d = len(set(self.ratings['item']))
        self.ratings=self.ratings[self.ratings['user'].isin(S)]
        reader=Reader(rating_scale=(1,5))
        data = Dataset.load_from_df(self.ratings[['user', 'item', 'rating']], reader)
        train_set=data.build_full_trainset()
        self.model.fit(train_set)
        
        self.inv_cate_dict={} #{'categoryA':[],'categoryB':[]}
        for i in self.category:
            self.inv_cate_dict[i]=[]
        for j in train_set.all_items():
            item_raw=train_set.to_raw_iid(j)
            self.inv_cate_dict[self.cate_dict_temp[item_raw]].append(j)
            self.price_dict[j]=self.price_dict_temp[item_raw]
            self.cate_dict[j]=self.cate_dict_temp[item_raw]
        self.cate_dict_temp.clear()
        self.price_dict_temp.clear()
        print("inv_cate_dict constructed.")
        d=0
        for i in self.category:
            d+=len(self.inv_cate_dict[i])
            print(i,':',len(self.inv_cate_dict[i]))
            
        self.ratings_predict=np.zeros([n,d])
        for i in train_set.all_users():
            user_raw=train_set.to_raw_uid(i)
            for j in train_set.all_items():
                item_raw=train_set.to_raw_iid(j)
                self.ratings_predict[i][j]=self.model.predict(user_raw, item_raw)[3]
        print("predicted ratings generated.")
        
        self.ranking=np.zeros([n,d])
        temp={}
        for i in range(n):
            for c in self.category:
                temp[c]=sorted(self.ratings_predict[i][self.inv_cate_dict[c]],reverse=True)
            for j in range(d):
                c=self.cate_dict[j]
                self.ranking[i][j]= temp[c].index(self.ratings_predict[i][j])+1
        print("user_item rankings generated.")
    
    def save_data(self,save_path):
        self.dict_all={'prices':self.price_dict,#'raw_ratings':self.ratings_matrix,
                           'new_ratings':self.ratings_predict,'cate':self.cate_dict,
                           'rankings': self.ranking,'max_price':self.max_price,
                           'new_price':self.new_price_dict}
                           #'user_mapper':self.user_mapper, 'item_mapper':self.item_mapper, 
                           #'user_inverse_mapper':self.user_inverse_mapper, 'item_inverse_mapper':self.item_inverse_mapper}
        with open(save_path,'wb') as f:
            pickle.dump(self.dict_all, f)
        print("data saved in ", save_path)
            
    def load(self,save_path):
        with open(save_path,'rb') as f:
            self.dict_all=pickle.load(f)
        #self.ratings_matrix =self.dict_all['raw_ratings']
        self.ratings_predict=self.dict_all['new_ratings']
        self.price_dict=self.dict_all['prices']
        self.cate_dict=self.dict_all['cate']
        self.ranking=self.dict_all['rankings']
        self.max_price=self.dict_all['max_price']
        self.new_price_dict=self.dict_all['new_price']
        #self.user_mapper=self.dict_all['user_mapper']
        #self.item_mapper=self.dict_all['item_mapper']
        #self.user_inverse_mapper=self.dict_all['user_inverse_mapper']
        #self.item_inverse_mapper=self.dict_all['item_inverse_mapper']
        self.dict_all.clear()
        del self.dict_all
        print("Saved data loaded.")
    
    def gen_new_price_dict(self):
        self.new_price_dict={}
        for i in self.category:
            self.new_price_dict[i]={}
        for i in range(len(self.cate_dict)):
            self.new_price_dict[self.cate_dict[i]][i]=self.price_dict[i]
        print("new price dictionary generated.")

class MBRecsys(object):
    def __init__(self,train_R,top_value):
        self.train_R=train_R
        self.top_value=top_value
        
    def predict(self):
        self.train_R
        U, s, VT = svds(self.train_R, k = self.top_value)  #select top 15 sigular value
        S=np.diag(s)
        self.out_R= np.dot(np.dot(U, S), VT)+3
        self.out_R= self.out_R- np.minimum(self.out_R,0)
        self.out_R= self.out_R+5- np.maximum(self.out_R,5)
        return self.out_R


def create_output():
    cat_names = ['Patio_Lawn_and_Garden','Musical_Instruments','Grocery_and_Gourmet_Food','Sports_and_Outdoors','Cell_Phones_and_Accessories']
    save_names = 'feature_1'
    d = DataLoader(cat_names,save_names)
    return d


if __name__ == '__main__':
    category=['Patio_Lawn_and_Garden','Musical_Instruments','Grocery_and_Gourmet_Food','Sports_and_Outdoors','Cell_Phones_and_Accessories']
    save_name='feature_1'
    d=DataLoader(category,save_name)
