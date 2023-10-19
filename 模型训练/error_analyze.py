from utils import compute_WMAPE
import pandas as pd


def delete_error_data_1(train_df,submission_df):
    """
    剔除造成误差大的数据
    :param train_df:  模型的中间训练数据（经过特征工程），比如："./中间数据/月度数据无编码_12.csv"
    :param submission_df: 已经初步训练过的提交（验证集）数据 ，比如"./提交数据/lgbm_submission.csv"
    :return:
    """
    df = submission_df
    df["error"] = abs(df["item_cnt_month_valid"] - df["item_cnt_month_pred"]) / df["item_cnt_month_valid"]
    df_38 = df[df["date_block_num"] == 41].sort_values(by="error", ascending=False)
    df_39 = df[df["date_block_num"] == 42].sort_values(by="error", ascending=False)

    # 筛选出误差比较大的 这些都是本来
    df_38_error = df_38[(df_38["error"]>10)&(df_38["item_cnt_month_pred"]>50)]
    df_39_error = df_39[(df_39["error"]>10)&(df_39["item_cnt_month_pred"]>50)]

    result_items_38 = set(zip(df_38_error['item_code'], df_38_error['sales_region_code']))
    result_items_39 = set(zip(df_39_error['item_code'], df_39_error['sales_region_code']))
    result_items = result_items_38.union(result_items_39)


    # 过滤数据
    item_codes = [item[0] for item in result_items]
    sales_region_codes = [item[1] for item in result_items]
    filtered_train_df = train_df[~(train_df['item_code'].isin(item_codes) & train_df['sales_region_code'].isin(sales_region_codes))]

    df = pd.DataFrame(result_items, columns=['item_code', 'sales_region_code'])
    df.to_csv("../中间数据/剔除数据1.csv",index=False)
    filtered_train_df.to_csv("../中间数据/月度数据无编码_12_剔除误差.csv",index=False)
    print("剔除以下商品和地区",result_items)
    print("已剔除离谱数据！")



def delete_error_data_2(train_df,submission_df):
    """
    剔除造成误差大的数据
    :param train_df:  模型的中间训练数据（经过特征工程），比如："./中间数据/月度数据无编码_12.csv"
    :param submission_df: 已经初步训练过的提交（验证集）数据 ，比如"./提交数据/lgbm_submission.csv"
    :return:
    """
    df = submission_df
    df["error"] = abs(df["item_cnt_month_valid"] - df["item_cnt_month_pred"]) / df["item_cnt_month_pred"]
    df_38 = df[df["date_block_num"] == 41].sort_values(by="error", ascending=False)
    df_39 = df[df["date_block_num"] == 42].sort_values(by="error", ascending=False)

    # 筛选出误差比较大的 这些都是本来
    df_38_error = df_38[(df_38["error"]>10)&(df_38["item_cnt_month_valid"]>50)]
    df_39_error = df_39[(df_39["error"]>10)&(df_39["item_cnt_month_valid"]>50)]

    result_items_38 = set(zip(df_38_error['item_code'], df_38_error['sales_region_code']))
    result_items_39 = set(zip(df_39_error['item_code'], df_39_error['sales_region_code']))
    result_items = result_items_38.union(result_items_39)


    # 过滤数据
    item_codes = [item[0] for item in result_items]
    sales_region_codes = [item[1] for item in result_items]
    filtered_train_df = train_df[~(train_df['item_code'].isin(item_codes) & train_df['sales_region_code'].isin(sales_region_codes))]

    df = pd.DataFrame(result_items, columns=['item_code', 'sales_region_code'])
    df.to_csv("../中间数据/剔除数据2.csv",index=False)
    filtered_train_df.to_csv("../中间数据/月度数据无编码_12_剔除误差.csv",index=False)
    print("剔除以下商品和地区",result_items)
    print("已剔除离谱数据！")



def final_delete_error_data_1(train_df,submission_df):
    """
    剔除造成误差大的数据
    :param train_df:  模型的中间训练数据（经过特征工程），比如："./中间数据/月度数据无编码_12.csv"
    :param submission_df: 已经初步训练过的提交（验证集）数据 ，比如"./提交数据/lgbm_submission.csv"
    :return:
    """
    df = submission_df
    df["error"] = abs(df["item_cnt_month_valid"] - df["item_cnt_month_pred"]) / df["item_cnt_month_valid"]
    df_39 = df[df["date_block_num"] == 42].sort_values(by="error", ascending=False)

    # 筛选出误差比较大的 这些都是本来
    df_39_error = df_39[(df_39["error"]>10)&(df_39["item_cnt_month_pred"]>50)]

    result_items = set(zip(df_39_error['item_code'], df_39_error['sales_region_code']))


    # 过滤数据
    item_codes = [item[0] for item in result_items]
    sales_region_codes = [item[1] for item in result_items]
    filtered_train_df = train_df[~(train_df['item_code'].isin(item_codes) & train_df['sales_region_code'].isin(sales_region_codes))]

    df = pd.DataFrame(result_items, columns=['item_code', 'sales_region_code'])
    df.to_csv("../中间数据/剔除数据1.csv",index=False)
    filtered_train_df.to_csv("../中间数据/月度数据无编码_12_剔除误差.csv",index=False)
    print("剔除以下商品和地区",result_items)
    print("已剔除离谱数据！")



def final_delete_error_data_2(train_df,submission_df):
    """
    剔除造成误差大的数据
    :param train_df:  模型的中间训练数据（经过特征工程），比如："./中间数据/月度数据无编码_12.csv"
    :param submission_df: 已经初步训练过的提交（验证集）数据 ，比如"./提交数据/lgbm_submission.csv"
    :return:
    """
    df = submission_df
    df["error"] = abs(df["item_cnt_month_valid"] - df["item_cnt_month_pred"]) / df["item_cnt_month_pred"]
    df_39 = df[df["date_block_num"] == 42].sort_values(by="error", ascending=False)

    # 筛选出误差比较大的 这些都是本来
    df_39_error = df_39[(df_39["error"]>10)&(df_39["item_cnt_month_valid"]>50)]

    result_items = set(zip(df_39_error['item_code'], df_39_error['sales_region_code']))


    # 过滤数据
    item_codes = [item[0] for item in result_items]
    sales_region_codes = [item[1] for item in result_items]
    filtered_train_df = train_df[~(train_df['item_code'].isin(item_codes) & train_df['sales_region_code'].isin(sales_region_codes))]

    df = pd.DataFrame(result_items, columns=['item_code', 'sales_region_code'])
    df.to_csv("../中间数据/剔除数据2.csv",index=False)
    filtered_train_df.to_csv("../中间数据/月度数据无编码_12_剔除误差.csv",index=False)
    print("剔除以下商品和地区",result_items)
    print("已剔除离谱数据！")