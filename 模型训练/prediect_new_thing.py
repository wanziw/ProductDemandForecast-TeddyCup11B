
def brand_new_train_test_split(data,valid_num):
    # 划分
    valid = valid_num  # 最终这里是38，前期可以是37
    X_train = data[data.date_block_num < valid].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < valid]['item_cnt_month']
    X_valid = data[data.date_block_num.isin([valid, valid + 1])].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num.isin([valid, valid + 1])]['item_cnt_month']
    X_test = data[data.date_block_num == valid + 2].drop(['item_cnt_month'], axis=1)

    Y_test = data[data.date_block_num == valid + 2]['item_cnt_month'] # 这个不一定有

    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test


from utils import train_test_split, create_submission, compute_tweedie, compute_submission_rmse_WMAPE, compute_WMAPE, \
     label_encoding, inverse_label_encoding, delete_some_item
from models import LGBM_train,LGBM_train_cv,LGBM_new_train
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from error_analyze import delete_error_data_1,delete_error_data_2

def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)

merge_error = 0 # 是否覆盖提交
valid_num = 38
# 误差分析
# raw_df = pd.read_csv("../中间数据/月度数据无编码_12.csv")
# sub_df = pd.read_csv("../提交数据/lgbm_submission.csv")
# delete_error_data_1(raw_df,sub_df)  # 筛除实际销量小的
# raw_df_2 = pd.read_csv("../中间数据/月度数据无编码_12_剔除误差.csv")
# delete_error_data_2(raw_df_2,sub_df)# 筛除实际销量大的 影响并不是很大

# 你先用月度数据无编码_12跑一遍，得到lgbm_submission，再用误差分析跑
df = pd.read_csv("../中间数据/新品月度数据无编码_12.csv")

# 删除不相关特征
# delete_cols = ["first_cate_code","second_cate_code","date_region_avg_item_cnt_lag_2","item_subtype_first_sale","month","date_region_avg_item_cnt_lag_6","date_avg_item_cnt_lag_1","date_region_avg_item_cnt_lag_12","days"]
# for i in delete_cols:
#     del df[i]

# label_encoding
LE1, LE2, LE3, LE4 = label_encoding(df)




X_train, Y_train, X_valid, Y_valid, X_test, Y_test = train_test_split(df,valid_num = valid_num)


# 交叉验证搜索
# model = LGBM_train_cv(X_train, Y_train)

# 选定最佳参数之后
model = LGBM_new_train(X_train, Y_train)


# 查看特征重要性
plot_features(model, (22,14))
# plt.show()
plt.savefig("../图片/新品特征重要性.png",dpi=200)
Y_valid_pred = model.predict(X_valid)  # 验证集的预测y
Y_test_pred = model.predict(X_test)  # 预测集的预测y

# 如果是验证集
Y_pred = Y_valid_pred
Y_valid = Y_valid

submission = create_submission(X_valid, Y_valid, Y_pred)
submission_full = pd.read_csv('../提交数据/stacking_rf_submission_new.csv')

# 覆盖提交
if merge_error==1:
    error_submission = pd.read_csv("../提交数据/lgbm_submission_error_item.csv")
    submission = pd.concat([submission,error_submission],axis=0)


# 还原编码
inverse_label_encoding(submission, LE1, LE2, LE3, LE4)

submission.to_csv(f'../提交数据/brand_new_lgbm_submission.csv', index=False)

# 计算一下训练集的指标
Y_train_pred = model.predict(X_train)
compute_WMAPE(Y_train, Y_train_pred)


# 计算一下验证集的指标
compute_tweedie(submission)
compute_submission_rmse_WMAPE(submission)


# second_layer_df.to_csv("../提交数据/模型融合.csv")