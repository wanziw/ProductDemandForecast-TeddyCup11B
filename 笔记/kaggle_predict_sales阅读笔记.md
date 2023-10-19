[toc]

- arima 时间序列预测
  - 消除数据平稳性
- 数据文件命名
- 去年科大讯飞商品预测

- 看看油管



- kaggle笔记
  - 其对shop、item类型、item进行了一系列预处理，这些在泰迪杯里都是没有的
- EDA
  - [Model stacking, feature engineering and EDA](https://www.kaggle.com/code/dimitreoliveira/model-stacking-feature-engineering-and-eda/notebook)
  - [Time series Basics : Exploring traditional TS](https://www.kaggle.com/code/kyakovlev/1st-place-solution-part-1-hands-on-data/notebook)




# 第一篇：中文的博客

- 粗略的过一遍，然后看第二篇，第二篇中有看不懂的可以再来这里找

- https://www.jianshu.com/p/1f6eef8a86fd

- 主要是下面这篇：

- [有详细的EDA和随机森林](https://www.jianshu.com/p/1f6eef8a86fd)
  - EDA

    - 删除异常值，会影响训练
    - 研究商品的上下架，test中是否存在train中没有的商品

  - 特征工程

    - 添加Lag features，滞后特征
    - Mean encoding（其实我是有一点没看懂的）
    - Price trend features（销量和售价呈反比，因此我们为每个样本增加的售价涨跌趋势特征（前一个月的））
    - Revenue trend features（validation set和test set都会出现train set所没有的新商品，借助商店revenue趋势特征可以帮助算法预测新商品在现有店铺的销量。泰迪杯没有商店的概念，可以用商品类别代替）
    - The resident or new item features
      - 'item_shop_last_sale'和'item_last_sale'用以纪录距离最近一次销售之间隔了几个月，通过它可以和之前月份的数据建立关联。'item_shop_first_sale'和'item_first_sale'则是用于表示新商品的特征

  - 训练

    - 将category数据数字化、填充NaN数据。用最后两个月的数据分别构建validation set和test set
    - 随机森林
    - 选取重要特征
    - 最后，有时间最好还是把valid set重新加入train set，并重新训练，我这里是为了节省时间所以没有重新训练模型

  - 其他

    - 你如果有兴趣，也可以用其他模型试试，如xgboost、LightGBM以及前作介绍的[深度神经网络](https://www.jianshu.com/p/f0d34d1952f0)，或者把它们的结果ensembling起来，这样预测的效果可能会更好



- [也是kaggle比赛的中文复盘](https://www.cnblogs.com/YeZzz/p/13574801.html)
- [kaggle时间序列常构建的五种时间特征](https://www.modb.pro/db/327355?utm_source=index_ai)

# 第二篇：好

自己从头到尾看完的，里面有中文注释

https://www.kaggle.com/code/ashleybryantsoejje/future-sales-3

写的很好，拿来做月度的预测已经完全没有问题了，但是这b泰迪杯还要求以天、周为精度的预测，搞不懂，可能无法这么复杂



一个小小的思路，将subtype和地区组合，相当于kaggle的shop

# 第三篇 粗略的特征分析

- [1st place solution - Part 1 - "Hands on Data"](https://www.kaggle.com/code/kyakovlev/1st-place-solution-part-1-hands-on-data/notebook)
- 比较粗略，想法并不是很成熟
- Item_id
  - 基础的通过修改数据格式，压缩数据内存
  - EDA
    - 每月sum、mean，分析一下训练集中有多少商品过期了（6个月没有销售），再看看这些商品在测试集中有多少

  - 异常值处理（根据价格和销量检测）
  - 可能的相关特征，跟item_id相关
    - lags
    - Release date（发布日期）
    - Last month sale
    - Days on sale
    - Neighbors (items with id 1000 and 1001 could be somehow similar - genre, type, release date)

- shop_id（在泰迪杯可以牵扯到产品分类看看）
  - 按shop_id分组，发现有些店刚开业时有过高峰，有些店已经过时（列举出每个月还没开的店，和已经过时的店）
  - 然后发现测试集中有一些是刚营业的店，而且没有过期店的数据
  - 可能的相关特征，跟shop_id相关
    - Lags (shop_id/shp_cnt_mth)
    - Opening month (possible opening sales)
    - Closed Month (possible stock elimination)

  - 迁移到泰迪杯就是哪个生产线在生产？哪个已经没有了？如果后续有对item_category分析的当我没说
- price
  - 特征
    - Price category (1/10/10/20$/ etc.) - obviously (or not obviously), items with smaller price have greater volumes
      - 价格和销量间的关系，一般来说价格便宜的东西销量高

    - Discount and Discount duration
      - 折扣

    - Price lag (shows discount)
    - Price correction (rubl/usd pair)
    - Shop Revenue

- dates
  - 相关特征
    - Weekends and holidays sales (to correct monthly sales)
      - 用该月的休息日和节假日营业额去修正月营业额

    - Number of days in the month (to correct monthly sales)
      - 用该月有多少天去修正月营业额

    - Month number (for seasonal items)
      - 对于季节性消费的产品重要，会影响

- item
  - item_id编码后的额外信息
  - mean coding
- Category
  - 分类
- 对test测试集的分析
  - 分析item是那些类型，是train中shop/item都对应的，还是啥也没有的，还是train中只有item没有shop的
  - 这样子的分析，是因为不同类型的item要采取不同的预测策略， 
  - "No Data Items" - it is more likely classification task



# 第四篇：还行

- [Feature engineering, xgboost](https://www.kaggle.com/code/dlarionov/feature-engineering-xgboost/notebook)
- 跟第二篇，大部分一样，可以结合起来考虑



# 第五篇： 

[Predict Future Sales Top 11 Solution](https://www.kaggle.com/code/szhou42/predict-future-sales-top-11-solution/notebook)

- 目前应该就是XGBoost、传统的ARIMA、跟神经网络的比较了
- 



# 第六篇：EDA

## 学习

[Model stacking, feature engineering and EDA](https://www.kaggle.com/code/dimitreoliveira/model-stacking-feature-engineering-and-eda/notebook)



- 可以学习一下一些可视化

  - How sales behaves along the year?
  - What category sells more?
  - What region sells more?
  - Feature "item_cnt" distribution.
  - Checking for outliers

- 特征这里好像有点问题，以下摘抄一些能借鉴的

  - 商品的价格与其（最低/最高）历史价格相比变化了多少
  - Rolling window based features (window = 3 months)(没看懂)
  - lag features
  - item sales count trend

- 训练集train、测试集validation划分

- 划分完之后做mean encoding

  - 对shop、item、shop/item、year、month，分别做销量的mean，作为mean encoding
  - 然后分别赋予train set和 validation set
  - 对啊，有些特征，你要划分完后，对训练集做，注意前面的那些博客有没有写错的地方

- 构建test

- modeling

  - 使用catboost、xgboost、random forest、knn、linear regression

  - > 使用一级模型的预测创建新的数据集。
    > 在这里，我将使用一种简单的组合技术，我将用第一级模型的预测作为第二级模型的输入，这样，第二级的模型将基本上使用第一级模型预测作为特征，并学习在哪里给予更多的权重。
    > 要使用这种技术，我还需要使用第一级模型并在测试集上进行预测，这样我就可以在第二级模型上使用它们。
    > 我还可以将带有额外特征的完整验证集（第一级模型预测）传递给第二级模型，让它在寻找解决方案方面做更多的工作。

  - ensembling

    - 用一个简单的线性回归，将5个模型的输出串起来



## 哥们自己做的可视化

- 分析价格和销量关系
  - 一个总体的：可以看出U形，说明对于总体来说，价格高的商品，销量相对少，对于价格低的商品，一半卖的多
  - 对于每种商品来说
    - 据观察，大部分商品，价格往往在一个区间内波动，偶尔会出现价格极高或极低的情况，价格极高的一半销量少
    - 粗略来看分为几种
      - 零星分布型：总共就没卖几次，很随机
      - 一条型
      - 多条型
    - 可以补充一下不同类别的一个先相关系数的计算，散点图上添加一条拟合线，让chatgpt做一下

- 分析线上线下
  - 可以补充线上/线下的商品需求量的变化波动情况

- 不同种类
  - 


# 第7篇：best score

- EDA
  - 检查一下region、item、catagory是不是都是一一对应的

- time feature

> 根据训练数据帧创建日和月分辨率时间特征，例如，自每件商品的第一次和最后一次销售以来的天数。
>
> 每个商品首次销售后的时间也用于创建每日平均销售额特征（“item_cnt_day_avg”），这可能有助于纠正不到一个月的商品的销售计数，因此在前一个月无法购买

- price feature
  - 算出每月的平均价格
  - 一个特征：



- 泰迪杯线上线下怎么理解，要分开建模吗，统计研究一下商品是全线上/全线下/一半一半
  - 抑或者，将其加入笛卡尔积？这感觉有点不太对
- 泰迪杯这里，可能还真需要地区和item做笛卡尔积，。。
- 集成学习
- 有些特殊的商品或者地区或者啥啥(进行误差分析，找出造成误差大的)，如果有的话，其干扰影响比较大，要特殊、单独处理



# 第八篇 科大讯飞练习



- 评价指标不能简单的用rmse
  - https://scikit-learn.org/stable/modules/model_evaluation.html
- 要把重心放在特征构建上，其他的像模型融合不用太复杂
- 多步预测
  - 递归多步预测：先用前n天的数据训练，预测day n+1，然后用前n+1天的数据训练，预测day n+2，然后用前n+2天的数据训练，预测day n+3
  - 直接多步预测：用前n天的数据训练，分别预测day n+1, day n+2, day n+3（这里预测第k天就不能用第k-1天和k-2天的数据）
- EDA
  - 每一个商品进行每日的数据图查看
  - 需要多与业务结合
  - 查看商品的销售是否有规律，我觉得这个可能跟线上线下有关，还有季节性等因素
- 取对数处理
  - 所有的销量加起来查看，然后再取对数查看，发现会平滑一些
- 添加特征
- 模型训练
  - lightGBM主要是最大深度
  - XGBoost是最大深度和叶子结点数
- 销量预测问题属于强业务逻辑的问题，需要对业务有所理解
  - 抖音->主播->网红特征->反应到数据上？
  - 我们这里可以根据商品的观察，这里是一个大型制造业企业，面向经销商的出货数据
- 数据的重要性要远远大于模型
  - 数据质量，数据量，数据版本迭代

## 一个kaggle上预测沃尔玛超市销量

- 按大类别和小类别，分别建立lightGBM模型。然后递归和直接预测都有，然后对一个商品有6个模型

- 对于突变

  - 取对数

- 设定随机种子

- > 1.按大类别和小类别分别构建模型
  > 2.递归+非递归
  > 3.时序交叉验证
  > 4.LightGBM, objective=tweedie（销量取对数之后可以用mse)
  >
  > 5.没有做后处理，没有使用魔法乘数（结果*0.99之类的 比较玄学 看效果提升没 但这个在公榜私榜差距容易大）

- LightGBM, objective=tweedie

- https://www.kaggle.com/competitions/m5-forecasting-accuracy/discussion/150614

- 比较稳扎稳打

- lstm 1dcnn 之类的也可以用（更注重数据量大小），结构化数据用lgbm效果更好

# 车流量预测

https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/39395



## 关于调参

不需要花太多时间



![](https://pic.imgdb.cn/item/642fa2d2a682492fcc138ba3.jpg)



- 取对数，就可以用rmse/mse
  - 不取对数要用另外一个
- 时序交叉验证
  - ![](https://pic.imgdb.cn/item/6431149e0d2dde57772f1727.jpg)
- 特征工程（波动率可以试试）
  - 对于'Close，我为多个滞后期准备了两个特征：该動间当前值与平均位的比率的对数，以以及当前位与某个时期前的值的比家的对数、对于这些，我以了所有货币的平均值（由于缺少数阳，没有进行加权平均）。此外，还准备了每种货币之间的差异和所有货币的平均位，結果，此功能似乎运行良好。
  - 根据业务背景，会有爆发销量的可能性，这种异常一般不能剔除（因为有些订单就是大，有些小，要根据商品来）
    - 一种思路是，一个版本剔除，一个版本不剔除，分别建模然后融合
    - 一个思路是，分商品（经常爆发和不爆发比较稳健的），或者说如果类别存在这种规律，就按类建模，否则要定义这种爆发性商品
    - 还有种（预处理方法），针对每个商品的销量，先取对数，然后做min-max scaling, 但是max按90%分位数取。
  - 关于这种异常数据的应对：
    - 如果存在周期性，那么把特征做好了，模型是能预测出来的。
    - 如果是非周期性的异常情况，那么在实际业务中必须要对当天的数据进行标注！而不能指望模型能预测出异常情况的发生。

## 系统梳理特征工程

![](https://pic.imgdb.cn/item/643118d50d2dde57773569d2.jpg)

商品爆发有没有周期性，可不可以构造相关的特征

异常值主要结合业务背景

交叉特征这里比较少



## 特征筛选

- 特征筛选方法很多，一般主要看特征重要性。

- 推荐使用后向特征选择，如果特征数量特别多，可以使用批量的后向特征选择。
- 根据特征重要性排序删除特征重要性低的特征，并对比删除前后的模型预测效果，根据效果决定是否删除。



## 模型融合

![](https://pic.imgdb.cn/item/64311cbb0d2dde57773bd609.jpg)



## 后处理

部分商品在训练集时间区间未尾销量特别高，导致测试集时间区间销量可能也很高
而评价指标是RMSE（MAPE指标更好一点），这样的评价指标会导致销量数据没做标准化处理的情况下，销量高的商品对评价指标影响很大
所以有效的提分方法就是找出哪些商品在训练集时间区间末尾销量特别高，然后修正这些商品的预测值。
注意：对于这种一榜制的比赛可以采取直接修正的方法，对于两榜的比赛，则需要优化模型，或处理异常数据，或增加特征，根据交叉验证的结果选择有效的方法。而本
次比赛数据量过少，且异常数据过多没有明显规律且缺之特征数据，所以交叉验证并不会起到有效的作用，只能选择手动修正预测结果。



![](https://pic.imgdb.cn/item/643122040d2dde5777426b4f.jpg)

- 把历史的销量和预测的画在一起，就能看出问题来了



# 第九篇 top 1% Predict Future Sales, features + LightGBM

https://www.kaggle.com/code/deinforcement/top-1-predict-future-sales-features-lightgbm#Predictive-words-in-item_name

# 其他小文章

- 可视化
  - [https://www.kaggle.com/code/gaetanlopez/how-to-make-clean-visualizations](https://www.kaggle.com/code/gaetanlopez/how-to-make-clean-visualizations)
  
  - https://www.kaggle.com/code/vikasukani/video-game-sales-eda-visualizations-ml-models
  
  - ```
    from plotly import express as px
    
    ```
- 异常值处理
  - [Outlier!!! The Silent Killer](https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer)
- 计算商品





|      |      |
| ---- | ---- |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |
|      |      |

