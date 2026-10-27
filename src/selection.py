import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def calc_basic_stats(X):
    # 1. Будуємо матрицю кореляцій (наскільки колонки схожі між собою від 0 до 1)
    corr_matrix = pd.DataFrame(X).corr().abs()
    
    # 2. Беремо лише верхній трикутник таблиці, щоб не порівнювати колонку саму з собою
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 3. Знаходимо назви колонок, у яких кореляція з іншими перевищує 0.85 (тобто вони майже однакові)
    high_corr_cols = [column for column in upper.columns if any(upper[column] > 0.85)]
    
    return high_corr_cols


def three_level_feature_selection(X_train, y_train, k_filter=30, k_wrapper=15):
    """Трирівневий конвеєр відбору ознак."""
    X_train_df = pd.DataFrame(X_train).copy()
    
    # 1. Filter Method (SelectKBest)
    filter_selector = SelectKBest(score_func=f_classif, k=min(k_filter, X_train_df.shape[1]))
    filter_selector.fit(X_train_df, y_train)
    filter_cols = X_train_df.columns[filter_selector.get_support()].tolist()
    X_filter = X_train_df[filter_cols]

    # 2. Hybrid Wrapper Method (mlxtend SequentialFeatureSelector)
    lr = LogisticRegression(max_iter=500, random_state=42)
    sfs = SFS(
        lr, 
        k_features=min(k_wrapper, len(filter_cols)), 
        forward=True, 
        floating=False, 
        scoring='roc_auc', 
        cv=3, 
        n_jobs=-1
    )
    sfs.fit(X_filter, y_train)
    wrapper_cols = list(sfs.k_feature_names_)
    X_wrapper = X_filter[wrapper_cols]

    # 3. Embedded Method (L1 Penalty / Lasso)
    lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    lasso.fit(X_wrapper, y_train)
    embedded_mask = lasso.coef_[0] != 0
    embedded_cols = [col for col, keep in zip(wrapper_cols, embedded_mask) if keep]
    
    if not embedded_cols:
        embedded_cols = wrapper_cols

    # 4. Tree-Based Permutation Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_df[embedded_cols], y_train)
    
    # Оцінюємо важливість кожної колонки шляхом її перемішування (permutation)
    perm_imp = permutation_importance(rf, X_train_df[embedded_cols], y_train, scoring='roc_auc', random_state=42)
    
    # Створюємо підсумкову таблицю та сортуємо за спаданням
    importance_df = pd.DataFrame({
        'feature': embedded_cols,
        'importance': perm_imp.importances_mean
    }).sort_values('importance', ascending=False)

    # Повертаємо відсортований список назв найкращих колонок
    return importance_df['feature'].tolist()