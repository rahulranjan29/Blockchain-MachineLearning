import pandas as pd 
import numpy as np
import itertools

import random
import pickle
    
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f) 

def get_all_address(single,multiple):
    a=[]
    for m in multiple:
        a.extend(m)
    a.extend(single)
    return a


def create_query(single,multiple):
    query=[]
    random.shuffle(multiple)
    random.shuffle(single)
    query.append(multiple[0])
    query.append(single[0:10])
    return query


def create_posneg_pairs(single,multiple):
    ppair=[]
    npair=[]
    p_count=0
    neg_count=0
    for i,m in enumerate(multiple):
        print(i)
        random.shuffle(m)
        
        pair=itertools.combinations(m,2)
        for j,p in enumerate(pair):
            if j>2000:
                break
            p=p+(1,)
            ppair.append(p)
        p_count=len(ppair)
        pseudo_neg=[]
        for j,m in enumerate(multiple):
            if j>20:
                break
            if j==i:
                continue
            else:
                random.shuffle(m)
                pseudo_neg.extend(m[0:5])
        for p_add in m:
            random.shuffle(single)
            random.shuffle(pseudo_neg)
            neg=single[0:6]
            pneg=pseudo_neg[0:4]
            ls=[(p_add,x,0) for x in neg]
            ls1=[(p_add,x,0) for x in pneg]
            npair.extend(ls)
            npair.extend(ls1)
        neg_count=len(npair)
        print('PCOUNT-->'+str(p_count))
        print('NCOUNT-->'+str(neg_count))
    return ppair,npair        



def main():
    add=load_obj('addresses')
    multiple=[]
    single=[]
    for k,v in add.items():
        if len(v)>1:
            multiple.append(v)
        else:
            single.append(k)
    q=create_query(single,multiple)
    save_obj(q,'query')       
    all_add=get_all_address(single,multiple)
    
    save_obj(all_add,'classifier_addresses')
    p,n=create_posneg_pairs(single,multiple)
    save_obj(p,'pospair')
    save_obj(n,'negpair')
    
    
if __name__=='__main__':
    main()

