
from utils import train_test_split, create_submission, compute_tweedie, compute_submission_rmse_WMAPE, compute_WMAPE, \
    label_encoding, inverse_label_encoding, delete_some_item, add_prophet, create_final_submission, \
    final_train_test_split
from models import error_item_model_train,LGBM_train_cv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from error_analyze import delete_error_data_1,delete_error_data_2

def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)


valid_num = 45


# 你先用月度数据无编码_12跑一遍，得到lgbm_submission，再用误差分析跑
if valid_num < 44:
    df = pd.read_csv(f"../中间数据/月度数据无编码_12.csv")
else:
    df = pd.read_csv(f"../中间数据/月度数据无编码_12_{valid_num}.csv")


error_item_list_1 = pd.read_csv("../中间数据/剔除数据1.csv")
error_item_list_2 = pd.read_csv("../中间数据/剔除数据2.csv")
error_item_list = pd.concat([error_item_list_1,error_item_list_2],axis=0)
# 将需要匹配的列组合成一个元组
error_item_tuples = list(error_item_list[['item_code', 'sales_region_code']].itertuples(index=False, name=None))

mask = df[['item_code', 'sales_region_code']].apply(tuple, axis=1).isin(error_item_tuples)
error_df = df.loc[mask]


# 商品分层
# item_info = pd.read_csv("../中间数据/商品分层.csv")
# df = delete_some_item(error_df,item_info)

# 添加趋势
# df = add_prophet(df)

del df["delta_cnt_lag"]
del df["delta_price_lag"]
df = error_df
# label_encoding
LE1, LE2, LE3, LE4 = label_encoding(df)

if valid_num < 43:
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = train_test_split(df,valid_num = valid_num)
else:
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = final_train_test_split(df, valid_num=valid_num)


# 交叉验证搜索
# model = model_train_cv(X_train, Y_train)

# 选定最佳参数之后
model = error_item_model_train(X_train, Y_train)


# 查看特征重要性
# plot_features(model, (22,14))
# plt.show()

Y_valid_pred = model.predict(X_valid)  # 验证集的预测y
Y_test_pred = model.predict(X_test)  # 预测集的预测y

# 如果是验证集
Y_pred = Y_valid_pred
Y_valid = Y_valid




if valid_num < 43:
    submission = create_submission(X_valid, Y_valid, Y_pred)

    # 还原编码
    inverse_label_encoding(submission, LE1, LE2, LE3, LE4)

    submission.to_csv(f'../提交数据/lgbm_submission_error_item.csv', index=False)

    # 计算一下训练集的指标
    Y_train_pred = model.predict(X_train)
    compute_WMAPE(Y_train, Y_train_pred)


    # 计算一下验证集的指标
    compute_tweedie(submission)
    compute_submission_rmse_WMAPE(submission)
else:
    submission = create_final_submission(X_valid, Y_pred)

    # 还原编码
    inverse_label_encoding(submission, LE1, LE2, LE3, LE4)

    submission.to_csv(f'../提交数据/final_lgbm_submission_error_item.csv', index=False)
    submission.to_csv(f'../提交数据/final_lgbm_submission_error_item_{valid_num}.csv', index=False)



# second_layer_df.to_csv("../提交数据/模型融合.csv")