import pandas as pd
beer = pd.read_csv('test_datas/Beer_data.txt',sep=',',header=None,
                   names=['c','calories','sodium','alcohol','cost'])
X = beer[['calories','sodium','alcohol','cost']]

# K-means
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)
print(km.labels_)
beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
beer.sort_values('cluster')
print(beer)

#查看分类出来的每个簇的平均值
print(beer.groupby('cluster').mean())

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
km = KMeans(n_clusters=3).fit(X_scaled)
beer['scaled_cluster'] = km.labels_

# DBSCAN
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=10,min_samples=2).fit(X)
beer['DBSCAN_cluster'] = db.labels_

# 轮廓系数
from sklearn import metrics
score_scaled =metrics.silhouette_score(X,beer.scaled_cluster)
score = metrics.silhouette_score(X,beer.cluster)
socre_dbs = metrics.silhouette_score(X,beer.DBSCAN_cluster)
print('K-means 无标准化 轮廓系数：',score)
print('K-means 标准化 轮廓系数：',score_scaled)
print('DBSCAN 轮廓系数',socre_dbs)
