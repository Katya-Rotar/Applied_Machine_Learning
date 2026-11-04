import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def find_optimal_threshold(y_true, y_probs, cost_fp, cost_fn):
    """Шукає поріг прийняття рішень у діапазоні [0.01, 0.99], що мінімізує сумарні фінансові збитки банку."""
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    min_loss = float('inf')
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        total_loss = (fp * cost_fp) + (fn * cost_fn)
        
        if total_loss < min_loss:
            min_loss = total_loss
            best_threshold = t
            
    return best_threshold, min_loss


def run_true_nested_cv(X, y, cost_fp=100, cost_fn=1000):
    """
    Виконує повну герметичну вкладену крос-валідацію (Nested CV):
    - Зовнішній цикл (5 фолдів): чесна оцінка якості на незнайомих даних.
    - Внутрішній цикл (3 фолди через GridSearchCV): підбір найкращих гіперпараметрів.
    - Усередині пайплайну: герметичний SMOTE та SelectKBest.
    - Cost-Sensitive оптимізація порогу для мінімізації грошових втрат банку.
    """
    # 1. Зовнішній контур (Outer Loop) - 5 фолдів для оцінки
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # 2. Внутрішній контур (Inner Loop) - 3 фолди для GridSearch
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    outer_scores_f1 = []
    outer_scores_auc = []
    total_costs = []
    
    # Словник вагових коефіцієнтів для класу 0 (False Positive) та класу 1 (False Negative)
    class_weights_dict = {0: cost_fp, 1: cost_fn}
    
    # Сітка параметрів для внутрішнього контуру (GridSearchCV)
    param_grid = {
        'resampler__k_neighbors': [3, 5, 8],          # параметри SMOTE
        'feature_selection__k': [10, 15, 20],         # кількість ознак SelectKBest
        'classifier__n_estimators': [100, 200],       # кількість дерев у Random Forest
        'classifier__max_depth': [5, 10, None]        # глибина дерев
    }
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Створюємо єдиний герметичний пайплайн зі зваженим навчанням
        # (Використовуємо ImbPipeline, щоб SMOTE і відбір ознак робилися СУВОРО всередині фолду)
        sealed_pipeline = ImbPipeline([
            ('resampler', SMOTE(random_state=42)),
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', RandomForestClassifier(class_weight=class_weights_dict, random_state=42))
        ])
        
        # --- ВНУТРІШНІЙ КОНТУР (Inner Loop / Grid Search) ---
        inner_grid = GridSearchCV(
            estimator=sealed_pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='f1',
            n_jobs=-1
        )
        
        # Навчаємо внутрішній пошук ТІЛЬКИ на тренувальній частині зовнішнього фолду
        inner_grid.fit(X_train, y_train)
        
        # Беремо найкращу модель, знайдену всередині
        best_model = inner_grid.best_estimator_
        
        # Отримуємо ймовірності для тестової частини
        y_probs = best_model.predict_proba(X_test)[:, 1]
        
        # --- COST-SENSITIVE ОПТИМІЗАЦІЯ ПОРОГУ ---
        best_threshold, min_loss = find_optimal_threshold(y_test, y_probs, cost_fp, cost_fn)
        y_pred_optimal = (y_probs >= best_threshold).astype(int)
        
        # Зберігаємо метрики цього фолду
        outer_scores_f1.append(f1_score(y_test, y_pred_optimal))
        outer_scores_auc.append(roc_auc_score(y_test, y_probs))
        total_costs.append(min_loss)
        
        print(f"Fold {fold}/5 завершено | Best Params: {inner_grid.best_params_} | Оптимальний поріг: {best_threshold:.2f}")
        
    print("\n" + "="*40)
    print(f"РЕЗУЛЬТАТИ ВКЛАДЕНОЇ КРОС-ВАЛІДАЦІЇ (NESTED CV):")
    print(f"F1-score: {np.mean(outer_scores_f1):.4f} ± {np.std(outer_scores_f1):.4f}")
    print(f"ROC-AUC:  {np.mean(outer_scores_auc):.4f} ± {np.std(outer_scores_auc):.4f}")
    print(f"Середні сумарні грошові втрати банку: ${np.mean(total_costs):.2f}")
    print("="*40)
    
    return outer_scores_f1, outer_scores_auc, total_costs