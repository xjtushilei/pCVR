import platform

import pickle
from sklearn.cluster import KMeans       #导入K-means算法包
from sklearn.datasets import make_blobs

n_samples = 50
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

print(X)
print(y)
# Incorrect number of clusters
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
print(y_pred)
# with open('user_app_dic.data', 'rb') as train_data_file:
#     user_app_dict = pickle.loads(train_data_file.read())
#
# print(len(user_app_dict))