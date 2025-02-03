from tslearn.metrics import cdist_dtw
import numpy as np
from scipy.stats import entropy
import pickle

def similarity(data,n=43660):
    distances = cdist_dtw(data,n_jobs=-1)
    vector_no_cero = distances[np.tril_indices(len(data), -1)]
    N = len(vector_no_cero)
    if N > 0:
        H_max = np.log2(n)
        H = entropy(vector_no_cero, base=2)
        H_normalizada = H / H_max
        return H_normalizada
    else:
        return False
    
with open('/data/samuelrt/kmedoids/K_chuva_6H.pkl','rb') as archivo:
    cluster = pickle.load(archivo)   

# for k in cluster.keys():
#     print(f'{k} : {similarity(cluster[k])}')
print(similarity(cluster))