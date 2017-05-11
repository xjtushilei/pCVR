# 第一届腾讯社交广告-高校算法设计大赛

计算广告是互联网最重要的商业模式之一，广告投放效果通常通过曝光、点击和转化各环节来衡量，大多数广告系统受广告效果数据回流的限制只能通过曝光或点击作为投放效果的衡量标准开展优化。腾讯社交广告( http://ads.tencent.com )发挥特有的用户识别和转化跟踪数据能力，帮助广告主跟踪广告投放后的转化效果，基于广告转化数据训练转化率预估模型(pCVR，Predicted Conversion Rate)，在广告排序中引入pCVR因子优化广告投放效果，提升ROI。 本题目以移动App广告为研究对象，预测App广告点击后被激活的概率：pCVR=P(conversion=1 | Ad,User,Context)，即给定广告、用户和上下文情况下广告被点击后发生激活的概率。


# 安装过的包

换源：-i https://pypi.tuna.tsinghua.edu.cn/simple

- keras
- tensorflow（ tensorflow-gpu）
- numpy
- scipy
- h5py
- matplotlib
- scikit-learn 
- scikit-image


# 结果统计 
- 总数：`3749528`
- label_1: `93262`
- label_0: `3656266`

显然是一个不平衡的数据集

# data文件含义
- get_data_v1 仅仅是训练数据的id，没有具体的获得数据
- get_data_v2 增加关联的数据（目前还没有app记录，以后版本可以考虑用户社团发现）
- get_data_v3 解决数据不平衡问题：采用复制数据方法，来提高数据平衡性

# 模型文件
注意：请使用对应版本的数据
- dense  全连接层
- lstm  lstm层

# 经验
- 损失函数用binary_crossentropy（又叫做logloss）
- 加入 `BatchNormalization` 和 `dropout` 解决了loss直接收敛的问题
- 考虑直接在数据层归一化。

# 结果
- lstm_v2_bak.py 的在迭代一千次的情况下，达到了loss：0.084，并且还没有完全收敛，改天可以继续迭代。