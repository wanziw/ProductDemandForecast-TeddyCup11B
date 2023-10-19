from utils import train_test_split, create_submission, compute_tweedie, compute_submission_rmse_WMAPE, compute_WMAPE, \
     label_encoding, inverse_label_encoding, delete_some_item
from models import LGBM_train,LGBM_train_cv,XGBoost_train,XGBoost_train_cv
import pandas as pd
import xgboost as xgb

if __name__ == '__main__':
     merge_error = 1  # 是否覆盖提交
     valid_num = 38
     df = pd.read_csv("../中间数据/月度数据无编码_12_剔除误差.csv")

     X_train, Y_train, X_valid, Y_valid, X_test, Y_test = train_test_split(df, valid_num=valid_num)

     # XGBoost_train_cv(X_train, Y_train, X_valid, Y_valid)

     model = XGBoost_train(X_train, Y_train)





     # 对测试数据进行预测
     dtest = xgb.DMatrix(X_valid, label=Y_valid)
     Y_pred = model.predict(dtest)

     submission = create_submission(X_valid, Y_valid, Y_pred)

     submission.to_csv(f'../提交数据/xgb_submission.csv', index=False)

     # 计算一下验证集的指标
     compute_tweedie(submission)
     compute_submission_rmse_WMAPE(submission)