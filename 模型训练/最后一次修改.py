

import pandas as pd
import numpy as np

def modify(second_cate_code,shan_name,month,magic_para):
    df = pd.read_excel("../最终提交数据/result2_1（第二版）.xlsx")


    if shan_name == 0:
        region_code = [101,102,103]
    else:
        region_code = [105]

    str = f"2019年{month}月预测需求量"
    mask = (df["sales_region_code"].isin(region_code))&(df["second_cate_code"]==second_cate_code)


    df.loc[df.index[mask], str] = df.loc[df.index[mask], str]*magic_para  # month-2刚好是对应的那一列
    df.to_excel("../最终提交数据/result2_1（第二版）.xlsx",index=False)

    shan_name = "线下" if shan_name==0 else "线下"
    print(f"{second_cate_code}类产品{shan_name}渠道，第{month}月，魔法系数:{magic_para}")

if __name__ == '__main__':
    second_cate_code = 401
    shan_name = 0  # 0为线下，1为线上
    month = 4
    magic_para = 1.04
    # 401
    modify(second_cate_code=401, shan_name=0, month=4, magic_para=1.1)
    modify(second_cate_code=401, shan_name=0, month=5, magic_para=1.1)
    modify(second_cate_code=401, shan_name=0, month=6, magic_para=1.1)

    modify(second_cate_code=401, shan_name=1, month=5, magic_para=1.3)

    # 402
    modify(second_cate_code=402 , shan_name=0, month=4, magic_para=1.04)
    modify(second_cate_code=402 , shan_name=0, month=5, magic_para=1.1)
    modify(second_cate_code=402 , shan_name=0, month=6, magic_para=1.04)

    modify(second_cate_code=402 , shan_name=1, month=5, magic_para=1.3)

    # 403
    modify(second_cate_code=403, shan_name=0, month=4, magic_para=1.04)
    modify(second_cate_code=403, shan_name=0, month=5, magic_para=1.1)

    modify(second_cate_code=403, shan_name=1, month=4, magic_para=0.75)
    modify(second_cate_code=403, shan_name=1, month=5, magic_para=2.5)
    modify(second_cate_code=403, shan_name=1, month=6, magic_para=0.9)

    # 404
    modify(second_cate_code=404, shan_name=0, month=4, magic_para=1.04)
    modify(second_cate_code=404, shan_name=0, month=5, magic_para=1.1)
    modify(second_cate_code=404, shan_name=0, month=6, magic_para=1.1)


    modify(second_cate_code=404, shan_name=1, month=4, magic_para=0.75)
    modify(second_cate_code=404, shan_name=1, month=5, magic_para=2.3)
    modify(second_cate_code=404, shan_name=1, month=6, magic_para=0.9)

    # 405
    modify(second_cate_code=405, shan_name=0, month=4, magic_para=0.9)
    modify(second_cate_code=405, shan_name=0, month=5, magic_para=1.1)
    modify(second_cate_code=405, shan_name=0, month=6, magic_para=1.1)

    modify(second_cate_code=405, shan_name=1, month=4, magic_para=0.75)
    modify(second_cate_code=405, shan_name=1, month=5, magic_para=2.3)
    modify(second_cate_code=405, shan_name=1, month=6, magic_para=0.9)

    # 407
    modify(second_cate_code=407, shan_name=0, month=4, magic_para=1.04)
    modify(second_cate_code=407, shan_name=0, month=5, magic_para=1.04)
    modify(second_cate_code=407, shan_name=0, month=6, magic_para=1.04)

    modify(second_cate_code=407, shan_name=1, month=4, magic_para=0.85)
    modify(second_cate_code=407, shan_name=1, month=5, magic_para=2.6)
    modify(second_cate_code=407, shan_name=1, month=6, magic_para=0.65)

    # 408
    modify(second_cate_code=408, shan_name=0, month=4, magic_para=1.04)
    modify(second_cate_code=408, shan_name=0, month=5, magic_para=1.2)
    modify(second_cate_code=408, shan_name=0, month=6, magic_para=1.2)

    modify(second_cate_code=408, shan_name=1, month=4, magic_para=0.9)
    modify(second_cate_code=408, shan_name=1, month=5, magic_para=2.6)
    modify(second_cate_code=408, shan_name=1, month=6, magic_para=0.65)

    # 410
    modify(second_cate_code=410, shan_name=0, month=4, magic_para=1.08)
    modify(second_cate_code=410, shan_name=0, month=5, magic_para=1.08)
    modify(second_cate_code=410, shan_name=0, month=6, magic_para=1.08)

    # 412
    modify(second_cate_code=412, shan_name=0, month=4, magic_para=1.08)
    modify(second_cate_code=412, shan_name=0, month=5, magic_para=1)
    modify(second_cate_code=412, shan_name=0, month=6, magic_para=2.1)

    modify(second_cate_code=412, shan_name=1, month=4, magic_para=1.08)
    modify(second_cate_code=412, shan_name=1, month=5, magic_para=1.3)
    modify(second_cate_code=412, shan_name=1, month=6, magic_para=0.75)



print(1)