import pandas as pd
import numpy as np


df1 = pd.read_excel("../最终提交数据/result2_1（第二版）.xlsx")
df2 = pd.read_csv("../示例数据/B题-测试数据/predict_sku2.csv")

a = (df1[["sales_region_code","item_code"]]==df2[["sales_region_code","item_code"]])

print( sum(a["item_code"]))