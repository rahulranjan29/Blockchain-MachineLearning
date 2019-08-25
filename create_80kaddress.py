# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:12:03 2019

@author: Rahul Verma
"""

import pandas as pd
import numpy as np
import pickle
    
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f) 
    
#*** Get all the user address which belong to the same user/entity/organsation***
column_names=['address_id','userid']
data=pd.read_csv('contraction.txt',sep='\t',names=column_names)

uniq=[]
for i in range(1,len(data)):
    add,user=data.iloc[i][0],data.iloc[i][1]
    if add!=user:
        uniq.append((add,user,i))    
    if i%100000==0:
        print(i)
np.save('uniq.npy',uniq) ## SAVE THE FILE FOR LATER USER
#******------------******


# ***** LOAD THE PREVIOUSLY CREATED FILE***
### FILTER 80000 user address which will be used to for feature extraction and training the model 

users=np.load('uniq.npy')
us=users[:,0]
s=np.where(us<80001)[0]  ## Fetch all the user id which belong to 1-80000
l=[]
for i in s:
    l.append(users[i])

np.save('80kadd.npy',l)

#******------------******


#***CREATE DICTIONARY OF THE FORMAT (KEY,VALUE):(USER ID 1,USER ID 1) FOR ALL THE USERS IDS WHO DO NOT HAVE ANY SIMILAR USER IDS
kadd=np.load('80kadd.npy')
k=list(kadd[:,0])
new_add={}
for i in range(1,80001):
    if i not in k:
        new_add.update({i:[i]})  
#******------------******
        
# *** TO THE ALREADY CREATED DICTIONARY, IF ANY TWO OR MORE ADDRESS ID BELONG TO SAME USER ID, UPDATE THE DICTONARY //
# FORMAT (KEY,VALUE):(USER ID 2,[USER ID2 , USER ID3242, USER ID22144]) AND SO ON   
        
for i in range(kadd.shape[0]):
    value,key=kadd[i][0],kadd[i][1]
    prev_val=new_add[key]
    prev_val.append(value)
    new_add.update({key:prev_val})

save_obj(new_add,'addresses')

#******------------******