from sklearn.cluster import KMeans
import torch
import numpy as np
import pickle
from sklearn import metrics
from sklearn.manifold import TSNE

def purity_score(y_true, y_pred):
	# reference: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


with open("experiment46-force-joint-env.p", "rb") as f:
    dict = pickle.load(f)

# mat_to_num = {'glass':0, 'ice':1, 'rubber':2, 'steel':3, 'wood':4}
mat_to_num = {'earth':0, 'moon':1, 'mars':2}

X = []
true_label = []
for k in dict:
	for vec in dict[k]:
		X.append(vec)
		true_label.append(mat_to_num[k])

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_

purity = purity_score(true_label, labels)

print(purity)

# X_embedded = TSNE(n_components=2).fit_transform(X)
# print(X_embedded.shape)














