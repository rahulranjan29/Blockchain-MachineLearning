# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:59:43 2019

@author: Rahul Verma
"""

import numpy as np 
import pandas as pd
import pickle
from multiprocessing.pool import Pool
    
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)





def get_values(transactions):
    if len(transactions)==0:
        return 0.0,0.0,0.0,0.0,0.0
    transactions=[x/100000000 for x in transactions ]
    m=np.mean(transactions)
    s=np.std(transactions)
    maxx=np.max(transactions)
    minn=np.min(transactions)
    total=np.sum(transactions)
    return m,s,maxx,minn,total

def get_features(list_add,degree,balance,txin,txout):
    
    feat_dict={}
    
    
    for i,address in enumerate(list_add):
        feat=[]
        index_degree=np.where(degree['address_id']==address)[0]
        
        index_balance=np.where(balance['address_id']==address)[0]
        if len(index_degree)==0 or len(index_balance)==0:
            print("skipped-->"+str(i))
            continue
        
        if len(index_degree)==0:
            print('Something wrong')
        index_txin=np.where(txin['input_add']==address)[0]
        index_txout=np.where(txout['output_add']==address)[0]
        
        feat.append(degree.iloc[int(index_degree[0])][2])
        feat.append(degree.iloc[index_degree[0]][1])
        feat.append(balance.iloc[index_balance[0]][1])
        
        transactions_input=[]
        transactions_output=[]
        for i in list(index_txin):
            transactions_input.append(txin.iloc[i][2])
        
        for i in list(index_txout):
            transactions_output.append(txout.iloc[i][2]) 
        
        
        m,s,maxx,minn,summ=get_values(transactions_input)
        m1,s1,maxx1,minn1,summ1=get_values(transactions_output)
        feat.append(m)
        feat.append(s)
        feat.append(maxx)
        feat.append(minn)
        feat.append(summ)
        feat.append(m1)
        feat.append(s1)
        feat.append(maxx1)
        feat.append(minn1)
        feat.append(summ1)
        
        feat_dict.update({address:feat})
        
    return feat_dict




def main():
    degree=pd.read_csv('degree.txt',sep='\t',names=['address_id','indeg','outdeg'])
    balance=pd.read_csv('balances.txt',sep='\t',names=['address_id','balance'])
    txin=pd.read_csv('txin.txt',sep='\t',names=['txId','input_add','input_value'])
    txout=pd.read_csv('txout.txt',sep='\t',names=['txId','output_add','output_value'])
    list_of_address=load_obj('classifier_addresses')
    print('processing')
    data=get_features(list_of_address,degree,balance,txin,txout)
    save_obj(data,'features')
  
if __name__ == '__main__':
    
    main()    
