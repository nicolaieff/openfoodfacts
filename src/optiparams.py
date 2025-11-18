import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import joblib

seed=21

def build_objective(X, y, X_out, y_out):
    def objective(trial):
        try:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0)
            }

            model = lgb.LGBMRegressor(**params, 
                                      random_state=seed
                                      )
            model.fit(X, y)
            
            y_pred = model.predict(X_out)
            if not np.all(np.isfinite(y_pred)):
                print("y_pred contains non-finite values!")
                raise optuna.exceptions.TrialPruned()
            
            sc = mean_squared_error(y_out, y_pred)

            if not np.isfinite(sc):
                raise optuna.exceptions.TrialPruned()

            return sc
        
        except Exception as e:
            print(f"[Trial {trial.number}] Failed with error: {e}")
            raise optuna.exceptions.TrialPruned()
    
    return objective


def optuna_search_lgb(X, y, X_out, y_out):
    objective = build_objective(X, y, X_out, y_out)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    return {"Best_params": study.best_params}


def optuna_search_knn(X, y, n_trials=30, path='./model/knn_model.joblib'):
    def objective(trial):

        n_neighbors = trial.suggest_int('n_neighbors', 3, 9, 2)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        p = trial.suggest_int('p', 1, 2)

        knn_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                p=p
                ))
                ])

        score = cross_val_score(knn_pipeline, X, y, cv=2, 
                                scoring='r2', 
                                n_jobs=-1
                                ).mean()
        return score

    study = optuna.create_study(direction='minimize')  # minimize, maximize
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = KNeighborsRegressor(**best_params)
    best_model.fit(X, y)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, path)

    return best_params, study.best_value


# ETAPE F : recherche de meilleurs parametres
# print('F. 🤖 KNN tuning')
# best_params, best_score = optuna_search_knn(X_all, y_all)

# # ETAPE G
# print('G. Save model')
# model_knn(X_all, y_all, save=True)

# study.best_trials[0].params
# trial.state

# Parcourir les trials
# print("Number of completed trials:", len(study.trials))
# for trial in study.trials:
#     print("Trial:", trial.number, "Value:", trial.value, "Params:", trial.params)
