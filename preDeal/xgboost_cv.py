import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

# 加载数据


print("开始反序列化加载数据...")
with open('get_data_v4.3.data', 'rb') as train_data_file:
    data = pickle.loads(train_data_file.read())
print("加载结束...")

(x, y, test_id, test) = data
# print(test_id[2])
# 前90%是训练数据，后10%是测试数据,通过反向传播来进行预测！
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print(len(x_train[1]))
print(y_train)

parameters = [
    {'learning_rate': [0.1, 0.3, 0.01]},
    {'max_depth': range(4, 15, 2)},
    {'subsample': [x / 10.0 for x in range(5, 10, 1)]}
]
clf = GridSearchCV(
    XGBClassifier(
        n_estimators=100,
        # max_depth=5,
        min_child_weight=1,
        gamma=0.5,
        # subsample=0.6,
        silent=0,
        colsample_bytree=0.6,
        objective='binary:logistic',  # 逻辑回归损失函数
        scale_pos_weight=1,
        reg_alpha=0,
        nthread=8,
        reg_lambda=1,
        seed=27),
    param_grid=parameters,
    scoring='roc_auc',
    verbose=1)
clf.fit(x_train, y_train)
print(clf.best_score_)
print(clf.best_params_)
y_pred = clf.predict(x_val)
print(log_loss(y_true=y_val, y_pred=y_pred))
y_pred = clf.predict(test)
print(y_pred)
