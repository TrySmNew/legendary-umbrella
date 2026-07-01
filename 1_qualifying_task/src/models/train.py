import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE


def metrics_report(model, X_test, y_test, show_plots=True):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print('Classification report:')
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f'ROC-AUC: {auc}')
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0, 0])
        axes[0, 0].plot([0, 1], '--')
        axes[0, 0].set_title('ROC Curve')
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[0, 1])
        axes[0, 1].set_title('Precision-Recall Curve')
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['0', '1'], yticklabels=['0', '1'], ax=axes[1, 0])
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 1].axis('off')
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    return {'roc_auc': auc, 'y_pred': y_pred, 'y_proba': y_proba}


def objective_catboost(trial, X, y):
    params = {
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'iterations': trial.suggest_categorical('iterations', [500, 1000, 1500]),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 50),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise']),
        'random_seed': RANDOM_STATE,
        'verbose': 0,
    }
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_ind, val_ind in skf.split(X, y):
        X_tr, X_val = X.iloc[train_ind], X.iloc[val_ind]
        y_tr, y_val = y.iloc[train_ind], y.iloc[val_ind]
        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, early_stopping_rounds=50, eval_set=(X_val, y_val))
        scores.append(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    return np.mean(scores)


def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [500, 1000, 1500]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
        'random_state': RANDOM_STATE,
        'eval_metric': 'auc',
        'n_jobs': -1,
    }
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_ind, val_ind in skf.split(X, y):
        model = XGBClassifier(**params)
        model.fit(X.iloc[train_ind], y.iloc[train_ind])
        scores.append(roc_auc_score(y.iloc[val_ind], model.predict_proba(X.iloc[val_ind])[:, 1]))
    return np.mean(scores)


def objective_lgbm(trial, X, y):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [500, 1000, 1500]),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'verbose': -1,
    }
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_ind, val_ind in skf.split(X, y):
        model = LGBMClassifier(**params)
        model.fit(X.iloc[train_ind], y.iloc[train_ind])
        scores.append(roc_auc_score(y.iloc[val_ind], model.predict_proba(X.iloc[val_ind])[:, 1]))
    return np.mean(scores)


def compute_oof_ensemble(X, y, catb_params, xgb_params, lgbm_params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_catb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_lgbm = np.zeros(len(X))
    best_iterations_catb = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m_catb = CatBoostClassifier(**catb_params, random_state=RANDOM_STATE, verbose=False, task_type='CPU', early_stopping_rounds=50, use_best_model=True)
        m_catb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_catb[val_idx] = m_catb.predict_proba(X_val)[:, 1]
        best_iterations_catb.append(m_catb.get_best_iteration())
        m_xgb = XGBClassifier(**xgb_params)
        m_xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        m_lgbm = LGBMClassifier(**lgbm_params)
        m_lgbm.fit(X_tr, y_tr)
        oof_lgbm[val_idx] = m_lgbm.predict_proba(X_val)[:, 1]
    oof_ensemble = (oof_catb + oof_xgb + oof_lgbm) / 3
    print(f'OOF CatBoost:  {roc_auc_score(y, oof_catb):.5f}')
    print(f'OOF XGBoost:   {roc_auc_score(y, oof_xgb):.5f}')
    print(f'OOF LightGBM:  {roc_auc_score(y, oof_lgbm):.5f}')
    print(f'OOF Ensemble:  {roc_auc_score(y, oof_ensemble):.5f}')
    return {'oof_catb': oof_catb, 'oof_xgb': oof_xgb, 'oof_lgbm': oof_lgbm, 'oof_ensemble': oof_ensemble, 'best_iterations_catb': best_iterations_catb}


def train_final_models(X_main, y_main, catb_params, xgb_params, lgbm_params, avg_best_iterations):
    ratio = (y_main == 0).sum() / (y_main == 1).sum()
    final_catb = catb_params.copy()
    final_catb['iterations'] = avg_best_iterations
    catb_model = CatBoostClassifier(**final_catb, random_state=RANDOM_STATE, verbose=False, task_type='CPU', scale_pos_weight=ratio)
    catb_model.fit(X_main, y_main, plot=True)
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_main, y_main)
    lgbm_model = LGBMClassifier(**lgbm_params)
    lgbm_model.fit(X_main, y_main)
    return catb_model, xgb_model, lgbm_model
