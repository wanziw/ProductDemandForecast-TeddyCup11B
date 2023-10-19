



import gc
import pickle

from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import pandas as pd
from matplotlib.pylab import rcParams
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_tweedie_deviance, mean_absolute_error, r2_score, make_scorer
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from lightgbm import plot_importance
import matplotlib.pyplot as plt

rcParams['figure.figsize'] = 12, 4

delete_cols = ['item_cnt_month_lag_1','item_cnt_month_lag_2','item_cnt_month_lag_3', 'date_avg_item_cnt_lag_1',
               'date_item_avg_item_cnt_lag_1','date_item_avg_item_cnt_lag_2','date_item_avg_item_cnt_lag_3','date_region_avg_item_cnt_lag_1',
               'date_region_avg_item_cnt_lag_2','date_region_avg_item_cnt_lag_3',
               'date_region_item_avg_item_cnt_lag_1','date_region_item_avg_item_cnt_lag_2',
               'date_region_item_avg_item_cnt_lag_3','date_region_subtype_avg_item_cnt_lag_1','delta_price_lag',
               'delta_revenue_lag_1', 'month', 'days', 'item_subtype_first_sale',
               'item_first_sale']
def train_test_split_M2(data,valid_num):
    # 划分
    valid = valid_num  # 最终这里是38，前期可以是37
    X_train = data[data.date_block_num < valid].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < valid]['item_cnt_month']
    X_valid = data[data.date_block_num.isin([valid, valid + 1])].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num.isin([valid, valid + 1])]['item_cnt_month']
    X_test = data[data.date_block_num == valid + 2].drop(['item_cnt_month'], axis=1)

    Y_test = data[data.date_block_num == valid + 2]['item_cnt_month']  # 这个不一定有

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test



def final_train_test_split(data,valid_num):
    # 划分
    valid = valid_num  # 最终这里是38，前期可以是37
    X_train = data[data.date_block_num < valid].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < valid]['item_cnt_month']
    X_valid = data[data.date_block_num == valid].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == valid]['item_cnt_month']
    X_test = data[data.date_block_num == valid + 1].drop(['item_cnt_month'], axis=1)

    Y_test = data[data.date_block_num == valid + 1]['item_cnt_month'] # 这个不一定有

    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

def train_test_split(data,valid_num):
    # 划分
    valid = valid_num  # 最终这里是38，前期可以是37
    X_train = data[data.date_block_num < valid].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < valid]['item_cnt_month']
    X_valid = data[data.date_block_num.isin([valid, valid + 1])].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num.isin([valid, valid + 1])]['item_cnt_month']
    X_test = data[data.date_block_num == valid + 2].drop(['item_cnt_month'], axis=1)

    Y_test = data[data.date_block_num == valid + 2]['item_cnt_month'] # 这个不一定有

    return X_train,Y_train,X_valid,Y_valid,X_test,Y_test

def train_model():


    pass


def create_submission(X_valid,Y_valid,Y_pred):
    submission = pd.DataFrame({
            # "ID": Y_valid.index,
        "date_block_num": X_valid.date_block_num,
        "sales_region_code": X_valid.sales_region_code,
        "item_code": X_valid.item_code,
        "first_cate_code": X_valid.first_cate_code,
        "second_cate_code": X_valid.second_cate_code,
        "item_cnt_month_valid": Y_valid,
        "item_cnt_month_pred": Y_pred
    })

    # 剔除所有item_cnt_month_valid为0的
    submission = submission[submission["item_cnt_month_valid"] != 0]
    # 填补负值
    submission["item_cnt_month_pred"] = list(map(lambda x: x if x > 0 else 0.1, submission["item_cnt_month_pred"]))

    return submission


def create_final_submission(X_valid,Y_pred):
    submission = pd.DataFrame({
            # "ID": Y_valid.index,
        "date_block_num": X_valid.date_block_num,
        "sales_region_code": X_valid.sales_region_code,
        "item_code": X_valid.item_code,
        "first_cate_code": X_valid.first_cate_code,
        "second_cate_code": X_valid.second_cate_code,
        "item_cnt_month_pred": Y_pred
    })


    # 填补负值
    submission["item_cnt_month_pred"] = list(map(lambda x: x if x > 0 else 0.1, submission["item_cnt_month_pred"]))

    return submission



def compute_tweedie(submission,code="",col=""):
    # 计算模型在两列数据上的Tweedie偏差
    tweedie_variance_power = 1.5
    tweedie_dev = mean_tweedie_deviance(submission["item_cnt_month_valid"], submission["item_cnt_month_pred"],
                                        power=tweedie_variance_power)

    # 打印Tweedie偏差值
    print("test:")
    print(f"{col}:{code}:Tweedie deviance:", tweedie_dev)

def compute_submission_rmse_WMAPE(submission,code="",col=""):
    print(f"{col}:{code}:rmse:", sqrt(mean_squared_error(submission["item_cnt_month_valid"], submission["item_cnt_month_pred"])))
    print(f"{col}:{code}:WMAPE",sum(abs(submission['item_cnt_month_valid'] - submission['item_cnt_month_pred'])) / sum(abs(submission['item_cnt_month_valid'])))



def compute_WMAPE(valid,predict,code="",col=""):
    print("train:")
    tweedie_variance_power = 1.6
    tweedie_dev = mean_tweedie_deviance(valid, predict,
                                        power=tweedie_variance_power)
    print(f"{col}:{code}:Tweedie deviance:", tweedie_dev)
    print(f"{col}:{code}:rmse:", sqrt(mean_squared_error(valid, predict)))
    print(f"{col}:{code}:WMAPE",sum(abs(valid - predict)) / sum(abs(valid)))




def add_series():

    pass

def label_encoding(sales_data):
    # 地区
    LE1 = LabelEncoder()
    sales_data["sales_region_code"] = LE1.fit_transform(sales_data.sales_region_code)
    # itemcode
    LE2 = LabelEncoder()
    sales_data["item_code"] = LE2.fit_transform(sales_data.item_code)
    # 大类
    LE3 = LabelEncoder()
    sales_data["first_cate_code"] = LE3.fit_transform(sales_data.first_cate_code)
    # 细类
    LE4 = LabelEncoder()
    sales_data["second_cate_code"] = LE4.fit_transform(sales_data.second_cate_code)
    return LE1,LE2,LE3,LE4;

def inverse_label_encoding(submission,LE1,LE2,LE3,LE4):
    # 还原编码
    submission["sales_region_code"] = LE1.inverse_transform(submission["sales_region_code"])
    submission["first_cate_code"] = LE3.inverse_transform(submission["first_cate_code"])
    submission["second_cate_code"] = LE4.inverse_transform(submission["second_cate_code"])
    submission["item_code"] = LE2.inverse_transform(submission["item_code"])
    return submission;


def delete_some_item(df, item_info):
    # 按照 item_code 合并 item_info 和 df
    df_merged = pd.merge(df, item_info, on='item_code')

    # 删除新品和流星品
    df_filtered = df_merged.query('item_type != "新品" and item_type != "流星品"')

    # 删除 item_type 列
    df_filtered = df_filtered.drop(columns='item_type')

    return df_filtered

def add_prophet(df):
    df2 = pd.read_excel("../中间数据/prophet月度趋势合成数据.xlsx")

    # 将 df1 的 ds 列转换为月编号
    df2['ds'] = pd.to_datetime(df2['ds'])
    df2['date_block_num'] = (df2['ds'].dt.year - 2015) * 12 + (df2['ds'].dt.month - 9)
    del df2['ds']
    del df2["first_cate_code"]
    del df2["second_cate_code"]
    # 合并 df1 和 df2
    df = pd.merge(df, df2, on=['date_block_num', 'sales_region_code', 'item_code'], how='left')
    return df




