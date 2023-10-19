
from utils import train_test_split,create_submission,compute_tweedie,compute_submission_rmse_WMAPE,compute_WMAPE,model_train,label_encoding,inverse_label_encoding
import pandas as pd
from utils import delete_cols

valid_num = 38

df = pd.read_csv("../中间数据/月度数据没编码.csv")
# label_encoding
LE1, LE2, LE3, LE4 = label_encoding(df)
# 尝试把训练集0的都清除进行训练？
# df_1 = df.loc[(df["date_block_num"] < valid_num) & (df["item_cnt_month"] != 0)].reset_index(drop=True)
# df_2 =  df.loc[df["date_block_num"] >= valid_num]
# df = pd.concat([df_1,df_2],axis=0)

second_layer_df = df.loc[:,["date_block_num","item_code","item_cnt_month","sales_region_code","first_cate_code","second_cate_code"]]
second_layer_df = second_layer_df[second_layer_df.date_block_num.isin([valid_num, valid_num + 1])]
second_layer_df = second_layer_df[second_layer_df.item_cnt_month!= 0]

varity_dicts = {"sales_region_code":[0,1,2,4],"first_cate_code":[0,1,2,3,4,5,6,7],"second_cate_code":[0,1,2,3,4,5,6,7,8,9,10,11]}
j = 0
for col,some_list in varity_dicts.items():
    for i in some_list:
        data = df[df[col] == i]
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = train_test_split(data,valid_num = valid_num)

        model = model_train(X_train, Y_train)


        Y_valid_pred = model.predict(X_valid)  # 验证集的预测y
        Y_test_pred = model.predict(X_test)  # 预测集的预测y

        # 如果是验证集
        Y_pred = Y_valid_pred
        Y_valid = Y_valid


        # 模型融合
        Y_pred_series = pd.Series(Y_pred, name=f'{col}_{i}_Y')
        X_valid_ = X_valid
        valid = pd.concat([X_valid_.reset_index(drop=True),Y_pred_series],axis=1)
        valid.drop(columns=delete_cols,inplace=True,axis=1)
        second_layer_df = pd.merge(second_layer_df,valid,on=['date_block_num', 'item_code',"sales_region_code","first_cate_code","second_cate_code"], how='left')



        submission = create_submission(X_valid, Y_valid, Y_pred)

        # 还原编码
        inverse_label_encoding(submission, LE1, LE2, LE3, LE4)

        # submission.to_csv(f'../提交数据/lgbm_submission_{col}_{i}.csv', index=False)

        # 计算一下训练集的指标
        Y_train_pred = model.predict(X_train)
        compute_WMAPE(Y_train, Y_train_pred, i, col)


        # 计算一下验证集的指标
        # compute_tweedie(submission,i,col)
        compute_submission_rmse_WMAPE(submission,i,col)

        j += 1
        print("-" * 10)

inverse_label_encoding(second_layer_df, LE1, LE2, LE3, LE4)
# second_layer_df.to_csv("../提交数据/模型融合.csv")