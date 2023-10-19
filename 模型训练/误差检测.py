from utils import compute_WMAPE
import pandas as pd
df = pd.read_excel("../中间数据/result1_直接.xlsx")


i = 3
Y_pred = df[f"2019年{i}月预测需求量"].fillna(0)
df[f"2019年{i}月真实需求量"] = list(map(lambda x: x if x > 0 else 0.1, df["2019年1月真实需求量"]))
Y = df[f"2019年{i}月真实需求量"]
compute_WMAPE(Y, Y_pred)