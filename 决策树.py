import matplotlib.pyplot as plt
import pandas as pd
# 房价数据集
from sklearn.datasets.california_housing import fetch_california_housing
housing = fetch_california_housing()

print(housing.data.shape)
print(housing.data[0])

from sklearn import tree
dtr = tree.DecisionTreeRegressor(max_depth=2)
dtr.fit(housing.data[:,[6,7]],housing.target)

dot_data = tree.export_graphviz(
    dtr,
    out_file=None,
    feature_names = housing.feature_names[6:8],
    filled = True,
    impurity=False,
    rounded = True
)
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor('#ff2DD')
graph.write_png('tree.png')

#参数
from sklearn.model_selection import train_test_split
data_train,data_test,target_train,target_test = \
train_test_split(housing.data,housing.target,test_size=0.1,
                 random_state=42)
dtr = tree.DecisionTreeRegressor(random_state=42)
dtr.fit(data_train,target_train)
print(dtr.score(data_test,target_test))  #0.637355881715626
#随机森林
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=42)
rfr.fit(data_train,target_train)
print(rfr.score(data_test,target_test)) #0.7910601348350835
from sklearn.model_selection import GridSearchCV
#交叉验证
# tree_parma_grid = {'min_samples_split':list((3,6,9)),
#                    'n_estimators':list((10,50,100))}
# gird = GridSearchCV(RandomForestRegressor(),param_grid=tree_parma_grid,cv=3)
# gird.fit(data_train,target_train)
# print(gird.best_params)
# print(gird.best_score)



