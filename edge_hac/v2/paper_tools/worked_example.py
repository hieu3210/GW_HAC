"""Table 4 of the paper (Iris, first split).  Run from edge_hac/v2:  python paper_tools/worked_example.py iris"""
import sys; sys.path.insert(0, '.')  # run from edge_hac/v2
import numpy as np, v2lib as L
from edgehac import datasets as D
from edgehac.whac import predict_words
from sklearn.metrics import adjusted_rand_score as ARI
name=sys.argv[1]
X,y=L.load_any(name,'dev'); K=len(np.unique(y))
tr,te=D.split(len(X),0); S_tr,S_te,keep=L.prepare_split(X,tr,te); F=S_tr.shape[1]; w=L.uniform_w(F)
cache={}; voc={l:L.family_vocab("HA-grid",l,S_tr,cache) for l in (1,2,3)}
print("theta,alpha", np.round(cache["grid"][0],2), np.round(cache["grid"][1],2))
f,_,q_tr,q_te,g=L.word_model(S_tr,S_te,voc[3],K,0,w)
print("N",len(X),"F",F,"K",K,"ntest",len(te))
for m in (0,1,2):
    gm=predict_words(q_te>>m,voc[3-m],f.P>>m,weights=w)
    print("level",3-m,"ARI",round(ARI(y[te],gm),3),"agree",round(np.mean(gm==g),3))
for k in range(K):
    for m in (0,1,2):
        print("cluster",k,"size",int(np.sum(g==k)),"lvl",3-m,[voc[3-m].names[j][f.P[k,j]>>m] for j in range(F)])
ms,per=L.certified_granularity(q_te,voc[3],f.P,w,3,lambda l:voc[l])
print("mstar dist",np.bincount(ms,minlength=3)/len(ms)); print(per)
print("majority class per cluster",[np.bincount(y[te][g==k]).argmax() if np.any(g==k) else -1 for k in range(K)])
