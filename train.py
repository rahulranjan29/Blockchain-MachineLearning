# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 22:38:37 2019

@author: Rahul Verma
"""
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import heapq

def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    


def get_query_features(feat,pairs):
    features=[]
    for i,pair in enumerate(pairs):
        idx1,idx2=pair[0],pair[1]
        x1,x2=feat[idx1],feat[idx2]
        X=np.multiply(x1, x2) / (np.linalg.norm(x1, 2) * np.linalg.norm(x2, 2))
        features.append(X)
        
    feature_matrix=np.vstack(features)
    
    return feature_matrix
def create_feature_matrix(feat,pairs):
    features=[]
    labels=[]
    for i,pair in enumerate(pairs):
        idx1,idx2,label=pair[0],pair[1],pair[2]
        print('processed-->'+str(i))
        if idx1 and idx2 in feat.keys(): 
            x1,x2=feat[idx1],feat[idx1]
        else:
            continue
        X=np.multiply(x1, x2) / (np.linalg.norm(x1, 2) * np.linalg.norm(x2, 2))
        features.append(X)
        labels.append(label)
    feature_matrix=np.vstack(features)
    x_train,x_test,y_train,y_test=train_test_split(feature_matrix, np.asarray(labels), train_size=0.7)
    return x_train,x_test,y_train,y_test



def cross_validate(X, y, clf, k=10):
    best_score, best_clf = 0.0, None
    kfold = KFold(k)
    for kid, (train, test) in enumerate(kfold.split(X, y)):
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        clf.fit(Xtrain, ytrain)
        ytest_ = clf.predict(Xtest)
        score = accuracy_score(ytest_, ytest)
        print("fold {:d}, score: {:.3f}".format(kid, score))
        if score > best_score:
            best_score = score
            best_clf = clf
    return best_clf, best_score

def test_report(clf, Xtest, ytest):
    ytest_ = clf.predict(Xtest)
    print("\n")
    print('***TEST REPORT***')
    print("\nAccuracy Score: {:.3f}".format(accuracy_score(ytest_, ytest)))
    print("\nConfusion Matrix")
    print(confusion_matrix(ytest_, ytest)) 
    print("\nClassification Report")
    print(classification_report(ytest_, ytest))

def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def main():
    feat=load_obj('features')    
    pos_pair=load_obj('pospair')
    neg_pair=load_obj('negpair')
    query=load_obj('query')
    pairs=[]
    pairs.extend(pos_pair)
    pairs.extend(neg_pair)
    x_train,x_test,y_train,y_test=create_feature_matrix(feat,pairs)
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-1) 
    best_clf, best_score = cross_validate(x_train, y_train, clf)
    test_report(best_clf, x_test, y_test)
    
    query_test=[]
    p=[(query[0][0],x) for x in query[0][1:]]
    n=[(query[0][0],x) for x in query[1]]
    query_test.extend(p),query_test.extend(n)
    clf_isotonic = CalibratedClassifierCV(best_clf, cv='prefit', method='sigmoid').fit(x_train,y_train)
    query_feat=get_query_features(feat,query_test)
    pred=clf_isotonic.predict_proba(query_feat)
    pred=pred[:,1]
    largest=heapq.nlargest(5,pred)
    largest=np.asarray(largest).reshape((5,1))
    index=np.where(np.asarray(pred)==largest)[1]
    k=unique(index)
    k=np.asarray(k)
    k=k[0:5]
    print("***Top 5 ranked nodes***")
    print("Format-->(Query Address,Similar Node)")
    for i in range(k.shape[0]):
        print("Degree of similarity:{},Rank:{}-->Nodes:{}".format(largest[i],i+1,query_test[k[i]]))
    


if __name__=="__main__": 
    main()