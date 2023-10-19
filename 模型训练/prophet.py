import pandas as pd
from utils import train_test_split, create_submission, compute_WMAPE, compute_tweedie, compute_submission_rmse_WMAPE

df1 = pd.read_excel("../中间数据/prophet月度合成数据.xlsx")
df2 = pd.read_csv("../中间数据/月度数据无编码_12.csv")
df2 = df2[["date_block_num","sales_region_code","item_code","item_cnt_month"]]



# 将 df1 的 ds 列转换为月编号
df1['ds'] = pd.to_datetime(df1['ds'])
df1['date_block_num'] = (df1['ds'].dt.year-2015) * 12 + (df1['ds'].dt.month-9)
del df1['ds']
df1["yhat"] = df1["yhat"]+0.5
# 合并 df1 和 df2
df = pd.merge(df1, df2, on=['date_block_num', 'sales_region_code', 'item_code'], how='right')
df.dropna(inplace=True)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = train_test_split(df,valid_num = 38)


submission = create_submission(X_valid, Y_valid, X_valid["yhat"])  # Y_pred
submission.to_csv(f'../提交数据/prophet_submission.csv', index=False)

# 计算一下训练集的指标

compute_WMAPE(Y_train, X_train["yhat"])


# 计算一下验证集的指标
compute_tweedie(submission)
compute_submission_rmse_WMAPE(submission)


print(1)


