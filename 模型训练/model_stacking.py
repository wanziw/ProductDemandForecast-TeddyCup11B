import numpy as np
import pandas as pd
from utils import train_test_split, create_submission, compute_tweedie, compute_submission_rmse_WMAPE, \
    label_encoding, inverse_label_encoding, delete_some_item, create_final_submission, final_train_test_split
from models import LGBM_train, LGBM_train_cv, XGBoost_train, CatBoost_train
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

merge_error = 0  # 是否覆盖提交
valid_num = 42
df = pd.read_csv("../中间数据/月度数据无编码_12.csv") #剔除误差
is_M = 1 # 1 2 3

# item_info = pd.read_csv("../中间数据/商品分层.csv")
# df = delete_some_item(df,item_info)

# M2滞后预测
if is_M == 2:
    df = pd.read_csv("../中间数据/M2月度数据无编码_12.csv")
    del df["item_cnt_month_lag_1"]
    df = df[df["date_block_num"]!=valid_num-1]

# M3滞后预测
if is_M == 3:
    df = pd.read_csv("../中间数据/M3月度数据无编码_12.csv")
    del df["item_cnt_month_lag_1"]
    del df["item_cnt_month_lag_2"]
    # df = df[df["date_block_num"]!=40]
    # df = df[df["date_block_num"]!=41]


if valid_num < 43:
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = train_test_split(df,valid_num = valid_num)
else:
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = final_train_test_split(df, valid_num=valid_num)





XGBoost = XGBoost_train(X_train, Y_train)
LGBM = LGBM_train(X_train, Y_train)
CatBoost = CatBoost_train(X_train, Y_train)

# 构建训练集和测试集的基础模型预测结果
train_base_predictions = []
test_base_predictions = []
for model in [LGBM, CatBoost]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_valid)
    train_base_predictions.append(train_pred)
    test_base_predictions.append(test_pred)

for model in [XGBoost]:
    dtest = xgb.DMatrix(X_valid, label=Y_valid)
    dtrain = xgb.DMatrix(X_train, label=Y_train)
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    train_base_predictions.append(train_pred)
    test_base_predictions.append(test_pred)

# 将基础模型的预测结果堆叠起来
train_stack = np.vstack(train_base_predictions).T
test_stack = np.vstack(test_base_predictions).T

# 构建随机森林模型
rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)

# 使用基础模型的预测结果来拟合随机森林模型
rf.fit(train_stack, Y_train)

# 得到随机森林模型的训练集和测试集的预测结果
train_pred_rf = rf.predict(train_stack)

test_pred_rf = rf.predict(test_stack)     # *1.04魔法乘数



if valid_num < 43:
    # 对测试数据进行预测
    submission = create_submission(X_valid, Y_valid, test_pred_rf)

    # 覆盖提交
    if merge_error == 1:
        error_submission = pd.read_csv("../提交数据/lgbm_submission_error_item.csv")
        submission = pd.concat([submission, error_submission], axis=0)

    submission.to_csv(f'../提交数据/stacking_rf_submission.csv', index=False)

    # 计算一下验证集的指标
    compute_tweedie(submission)
    compute_submission_rmse_WMAPE(submission)

else:
    submission = create_final_submission(X_valid, test_pred_rf)

    # 还原编码
    # inverse_label_encoding(submission, LE1, LE2, LE3, LE4)
    # 覆盖提交
    if merge_error == 1:
        error_submission = pd.read_csv("../提交数据/final_lgbm_submission_error_item.csv")
        submission = pd.concat([submission, error_submission], axis=0)

    submission.to_csv(f'../提交数据/final_stacking_rf_submission_{valid_num}.csv', index=False)


