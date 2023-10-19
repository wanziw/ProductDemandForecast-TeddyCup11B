import pandas as pd





test = pd.read_csv( "../示例数据/B题-测试数据/predict_sku2.csv")
# 直接预测
sub40 = pd.read_csv('../提交数据/final_lgbm_submission_43_0.csv')
sub41 = pd.read_csv('../提交数据/final_lgbm_submission_44_0.csv')
sub42 = pd.read_csv('../提交数据/final_lgbm_submission_45_0.csv')




# 滞后预测


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

merged_data.to_excel("../最终提交数据/result2_1.xlsx",index=False)
# 记得乘魔法系数1.1
print(1)