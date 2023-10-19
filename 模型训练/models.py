from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from lightgbm import plot_importance
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostRegressor

def LGBM_train(X_train, Y_train):
    # 参数
    learning_rate = 0.034
    max_depth = 30
    num_leaves = 140
    # 训练
    model = LGBMRegressor(learning_rate=learning_rate,
                          objective='tweedie',
                          max_depth=max_depth,
                          num_leaves=num_leaves,
                          n_jobs=-1,
                          n_estimators=300)

    model.fit(X_train, Y_train)


    return model

def LGBM_new_train(X_train, Y_train):
    # 参数
    learning_rate = 0.034
    max_depth = 20
    num_leaves = 30
    # 训练
    model = LGBMRegressor(learning_rate=learning_rate,
                          objective='tweedie',
                          max_depth=max_depth,
                          num_leaves=num_leaves,
                          n_jobs=-1,
                          n_estimators=300)

    model.fit(X_train, Y_train)


    return model

def error_item_model_train(X_train, Y_train):
    # 定义模型
    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=4,
        subsample=0.7,
        colsample_bylevel=1,
        loss_function='Tweedie:variance_power=1.5'
    )

    # 训练模型
    model.fit(X_train, Y_train, verbose=False)

    return model
    # # 参数
    # learning_rate = 0.04
    # max_depth = 10
    # num_leaves = 20
    # # 训练
    # model = LGBMRegressor(learning_rate=learning_rate,
    #                       objective='tweedie',
    #                       max_depth=max_depth,
    #                       num_leaves=num_leaves,
    #                       n_jobs=-1,
    #                       n_estimators=300)
    #
    # model.fit(X_train, Y_train)
    #
    #
    # return model

def LGBM_train_cv(X_train, Y_train):
    parameters = {
        'max_depth': range(20, 80, 5),  # max_depth ：设置树深度，深度越大可能过拟合
        'num_leaves': range(30, 170, 30),
        # num_leaves：因为 LightGBM 使用的是 leaf-wise 的算法，因此在调节树的复杂程度时，使用的是 num_leaves 而不是 max_depth。
        # 大致换算关系：num_leaves = 2^(max_depth)，但是它的值的设置应该小于 2^(max_depth)，否则可能会导致过拟合。

        # 这是提高精确度的最重要的参数

    }

    # scoring = {'MAE': make_scorer(mean_absolute_error), 'R2': make_scorer(r2_score)}

    gbm = LGBMRegressor(objective='tweedie',  # 回归 设置
                            learning_rate=0.3,
                            verbose=-1  ## <0 显示致命的, =0 显示错误 (警告), >0 显示信息
                            )
    gsearch = GridSearchCV(gbm, param_grid=parameters, cv=3) # scoring=scoring,
    gsearch.fit(X_train, Y_train)

    print('参数的最佳取值:{0}'.format(gsearch.best_params_))
    print('最佳模型得分:{0}'.format(gsearch.best_score_))
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.cv_results_['params'])




def XGBoost_train_cv(X_train,Y_train,X_valid,Y_valid):

     # 读取数据
     dtrain = xgb.DMatrix(X_train, label=Y_train)
     dtest = xgb.DMatrix(X_valid, label=Y_valid)

     # 定义参数空间
     params = {
          'objective': ['reg:tweedie'],
          'eval_metric': ['tweedie-nloglik@1.5'],
          'tweedie_variance_power': [1.5],
          'max_depth': [3,4,5],
          'eta': [0.1],
          'subsample': [0.5,0.6,0.7],
          'colsample_bytree': [0.8,0.9,1.0]
     }

     # 定义模型
     xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)

     # 进行网格搜索
     grid_search = GridSearchCV(xgb_model, params, cv=3, n_jobs=-1, verbose=3)
     grid_search.fit(X_train, Y_train)

     # 输出最优参数和得分
     print('Best parameters:', grid_search.best_params_)
     print('Best score:', grid_search.best_score_)

     # 使用最优参数重新训练模型
     # best_params = grid_search.best_params_
     # model = xgb.train(best_params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])




def XGBoost_train(X_train, Y_train):

     # 读取数据
     dtrain = xgb.DMatrix(X_train, label=Y_train)
     # dtest = xgb.DMatrix(X_valid, label=Y_valid)

     # 定义参数


     params = {
          'objective': 'reg:tweedie',
          'eval_metric': 'tweedie-nloglik@1.5',  # 使用Tweedie偏差评估指标
          'tweedie_variance_power': 1.5,  # Tweedie偏差的参数，可以根据数据集的特征进行调整
          'max_depth': 4,
          'eta': 0.1,
          'subsample': 0.6,
          'colsample_bytree': 0.9,
          'verbosity': 0
     }
     # # 训练模型
     model = xgb.train(params, dtrain, num_boost_round=100) # , evals=[(dtest, 'test')]
     return model


def CatBoost_train(X_train, Y_train):
    # 定义模型
    model = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=4,
        subsample=0.7,
        colsample_bylevel=1,
        loss_function='Tweedie:variance_power=1.5'
    )

    # 训练模型
    model.fit(X_train, Y_train, verbose=False)

    return model



def CatBoost_train_cv(X_train, Y_train, X_valid, Y_valid):
    # 定义参数空间
    params = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.5],
        'depth': [3, 4, 5],
        'subsample': [0.5, 0.6, 0.7],
        'colsample_bylevel': [0.8, 0.9, 1.0],
        'loss_function': ['Tweedie:variance_power=1.5']
    }

    # 定义模型
    model = CatBoostRegressor(random_state=42, verbose=False)

    # 进行网格搜索
    grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1, verbose=3)
    grid_search.fit(X_train, Y_train, eval_set=(X_valid, Y_valid))

    # 输出最优参数和得分
    print('Best parameters:', grid_search.best_params_)
    print('Best score:', grid_search.best_score_)

    # 返回最优模型
    return grid_search.best_estimator_
