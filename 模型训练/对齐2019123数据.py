import pandas as pd



# st = "滚动"   # 滞后
st = "直接"
test = pd.read_csv("../中间数据/月度数据无编码_12.csv")
data = pd.read_excel("../最终提交数据/result1.xlsx")
test = test[["date_block_num","sales_region_code","item_code","item_cnt_month"]]
# 直接预测
if st =="滚动":
    # sub40 = data["2019年1月预测需求量"]
    # sub41 = data["2019年2月预测需求量"]
    # sub42 =  data["2019年3月预测需求量"]
    sub40 = test[test["date_block_num"]==40]
    sub41 = test[test["date_block_num"]==41]
    sub42 = test[test["date_block_num"]==42]



    # 滞后预测


    merged_data = data.merge(sub40[['sales_region_code', 'item_code', 'item_cnt_month']],
                             on=['sales_region_code', 'item_code'], how='left')
    merged_data = merged_data.merge(sub41[['sales_region_code', 'item_code', 'item_cnt_month']],
                                    on=['sales_region_code', 'item_code'], how='left')
    merged_data = merged_data.merge(sub42[['sales_region_code', 'item_code', 'item_cnt_month']],
                                    on=['sales_region_code', 'item_code'], how='left')
    merged_data = merged_data.rename(columns={'item_cnt_month_x': '2019年1月真实需求量',
                                              'item_cnt_month_y': '2019年2月真实需求量',
                                              'item_cnt_month': '2019年3月真实需求量'})
    # del merged_data['first_cate_code']
    # del merged_data['second_cate_code']

    merged_data.to_excel("../中间数据/result1_直接.xlsx", index=False)
else:
    # 滞后预测
    sub40 = pd.read_csv('../提交数据/final_lgbm_submission_43_0.csv')
    sub41 = pd.read_csv('../提交数据/final_lgbm_submission_44_2.csv')
    sub42 = pd.read_csv('../提交数据/final_lgbm_submission_45_3.csv')



    merged_data = test.merge(sub40[['sales_region_code', 'item_code', 'item_cnt_month_pred']],
                             on=['sales_region_code', 'item_code'], how='left')
    merged_data = merged_data.merge(sub41[['sales_region_code', 'item_code', 'item_cnt_month_pred']],
                                    on=['sales_region_code', 'item_code'], how='left')
    merged_data = merged_data.merge(sub42[['sales_region_code', 'item_code', 'item_cnt_month_pred']],
                                    on=['sales_region_code', 'item_code'], how='left')
    merged_data = merged_data.rename(columns={'item_cnt_month_pred_x': '2019年4月预测需求量',
                                              'item_cnt_month_pred_y': '2019年5月预测需求量',
                                              'item_cnt_month_pred': '2019年6月预测需求量'})
    # del merged_data['first_cate_code']
    # del merged_data['second_cate_code']




    merged_data.to_excel("../最终提交数据/result1_滞后.xlsx",index=False)
# 记得乘魔法系数1.1
print(1)