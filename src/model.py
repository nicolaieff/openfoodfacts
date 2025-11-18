from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import cross_val_score

from lightgbm import LGBMRegressor
from pathlib import Path
from joblib import dump
import datetime

import numpy as np

seed = 21
cv = 2


def metrics(y, y_pred):

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    return {'MAE': f'{mae:.2f}',
            'RMSE': f'{rmse:.2f}',
            'R²': f'{r2:.3f}'}


def model_knn(X, y):
    
    params = {'n_neighbors': 8, 'weights': 'distance', 'p': 1}
    
    knn_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(
            **params
            ))
    ])

    knn_scores = cross_val_score(knn_pipeline, X, y, cv=cv, scoring="r2")
    knn_scores = np.round(knn_scores, 3)
    sc = round(np.mean(knn_scores), 3)  # scores.mean().round(3)
    return {
        'cv_sc': knn_scores, 
        'sc_mean': sc
        }


def model_tree(X, y):
    
    tree_pipeline = Pipeline(
        [("regressor", DecisionTreeRegressor(max_depth=5, random_state=seed))]
        )
    
    tree_scores = cross_val_score(tree_pipeline, X, y, cv=cv, scoring="r2")
    tree_scores = np.round(tree_scores, 3)
    sc = round(np.mean(tree_scores), 3)
    return {
        'cv_sc': tree_scores,
        'sc_mean': sc
        }

from sklearn.preprocessing import FunctionTransformer
def to_numpy(X):
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.array(X)


def model_lgb(X, y, params={}, save=False):
    params.pop("feature_name", None)

    lgb_pipeline = Pipeline([
        ("to_numpy", FunctionTransformer(to_numpy)),
        ("lgb", LGBMRegressor(
                              **params,
                              random_state=seed,
                            #   feature_name='auto',
                              verbose=-1
                              ))
    ])
    
    lgb_scores = cross_val_score(lgb_pipeline, X, y, cv=cv, scoring='r2')
    lgb_scores = np.round(lgb_scores, 3)
    sc = round(np.mean(lgb_scores), 3)

    if save:
        lgb_pipeline.fit(X, y)
        t = datetime.datetime.today().strftime('%Y_%m_%d')
        dump(lgb_pipeline, f'./model/lgb_pipeline_{t}.joblib')
        print('🤖 model save !')

    return {
        'cv_sc': lgb_scores,
        'sc_mean': sc
    }


def model_lin(X, y):

    pipeline = Pipeline([
        ("regressor", LinearRegression())
    ])

    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
    sc = round(np.mean(scores), 2)
    return {'sc': scores, 'sc_mean': sc}


# def search_bestparam(X, y, model, param_grid):
#     grid = GridSearchCV(model, 
#                         param_grid, 
#                         cv=cv, 
#                         scoring='r2'
#                         )
#     grid.fit(X, y)
#     return grid


# print('TREE model', model_tree(X, y, X_out, y_out))
# TREE model {'MAE': '4.26', 'RMSE': '5.82', 'R²': '0.679'}
# SANS PROCESS
# TREE model {'sc': array([0.6802249 , 0.67906559, 0.68067849, 0.68010241, 0.6783896 ]), 'sc_mean': np.float64(0.68)}
# AVEC LE PROCESS de knn
# TREE model {'sc': array([0.5566073 , 0.55914492, 0.55945422, 0.60485855, 0.6771424 ]), 'sc_mean': np.float64(0.59)}
# ⭐ DATA CLEAN -> dorp null ~9k data
# TREE model {'sc': array([0.77091244, 0.79233955, 0.79514296, 0.78427346, 0.7758406 ]), 'sc_mean': np.float64(0.78)}
# ⭐ DATA CLEAN -> null = 0
# TREE model {'sc': array([0.73192618, 0.73410949, 0.73725143, 0.73436063, 0.73585174]), 'sc_mean': np.float64(0.73)}

# print('KNN model', model_knn(X, y, X_out, y_out))
# KNN model {'MAE': '4.15', 'RMSE': '6.19', 'R²': '0.636'}
# KNN model {'sc': array([0.61574219, 0.70591547, 0.62031423, 0.70669371, 0.7138262 ]), 'sc_mean': np.float64(0.67)}
# ⭐ DATA CLEAN = dorp null ~9k data
# KNN model {'sc': array([0.90501206, 0.88325181, 0.8751897 , 0.90173345, 0.90368131]), 'sc_mean': np.float64(0.89)}
# ⭐ DATA CLEAN -> null = 0
# KNN model {'sc': array([0.88796902, 0.88892648, 0.88996251, 0.88632569, 0.88968759]), 'sc_mean': np.float64(0.89)}

# print('LINEAR model', model_lin(X, y))
# LINEAR model {'sc': array([0.57583665, 0.57990525, 0.57626358, 0.5760306 , 0.57504131]), 'sc_mean': np.float64(0.58)}

# Apres filtrage des boissons 
# 🏓 Score {'TREE': {'sc': array([0.75382723, 0.75506882, 0.75535827, 0.75184056, 0.75644317]), 'sc_mean': np.float64(0.75)}, 
# 'KNN': {'sc': array([0.90212974, 0.89972855, 0.89949147, 0.89936107, 0.90071171]), 'sc_mean': np.float64(0.9)}}

# LGBM sans parametre
# 'LGBM': {'cv_sc': array([0.899, 0.897, 0.896, 0.896, 0.899]), 'sc_mean': np.float64(0.897)}}

# LGBM apres optimisation de parametre
# LGBM  {'cv_sc': array([0.915, 0.913, 0.912, 0.912, 0.914]), 'sc_mean': np.float64(0.913)}
# LGBM  {'cv_sc': array([0.921, 0.919, 0.919, 0.918, 0.92 ]), 'sc_mean': np.float64(0.919)}

# + la feature fats, nuts, seeds
# TREE  {'cv_sc': array([0.754, 0.752]), 'sc_mean': np.float64(0.753)}
# KNN  {'cv_sc': array([0.91 , 0.909]), 'sc_mean': np.float64(0.91)}
# LGBM  {'cv_sc': array([0.901, 0.901]), 'sc_mean': np.float64(0.901)}

# feat KCAL revise
# TREE  {'cv_sc': array([0.754, 0.751]), 'sc_mean': np.float64(0.752)}
# KNN  {'cv_sc': array([0.91 , 0.906]), 'sc_mean': np.float64(0.908)}
# LGBM  {'cv_sc': array([0.902, 0.903]), 'sc_mean': np.float64(0.903)}

# avec la feature CAT - tag_0
# G. {'cv_sc': array([0.935, 0.934]), 'sc_mean': np.float64(0.935)}

# avec la feat tag2_0
# G. {'cv_sc': array([0.945, 0.944]), 'sc_mean': np.float64(0.944)}



# Note model
# preprocessing = [
#     ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
#     ("scaler", StandardScaler())
# ]
# reg_tree = DecisionTreeRegressor(max_depth=5, random_state=seed)
# reg_tree.fit(X, y)
# tree_pred = reg_tree.predict(X_out)
# metric = metrics(y_out, tree_pred)

    
# cross val 
# r2
# neg_mean_squared_error
# neg_root_mean_squared_error
# neg_mean_absolute_error
