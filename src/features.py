import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class CyclicFeatureEncoder(BaseEstimator, TransformerMixin):
    """Категорія А: Циклічне кодування (sin/cos) для часових ознак."""
    def __init__(self, cyclic_cols=None, max_val_dict=None):
        self.cyclic_cols = cyclic_cols or []
        self.max_val_dict = max_val_dict or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        for col in self.cyclic_cols:
            if col in X_out.columns:
                max_val = self.max_val_dict.get(col, X_out[col].max())
                X_out[f'{col}_sin'] = np.sin(2 * np.pi * X_out[col] / max_val)
                X_out[f'{col}_cos'] = np.cos(2 * np.pi * X_out[col] / max_val)
        return X_out


class FeatureInteractionTransformer(BaseEstimator, TransformerMixin):
    """Категорія Б: Взаємодії та відношення фінансових показників."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        
        # 1. Множимо завантаженість кредиток на коефіцієнт боргу (тобто клієнт немає вільних лімітів, і всі доходи йдуть на виплату боргів)
        if 'RevolvingUtilizationOfUnsecuredLines' in X_out.columns and 'DebtRatio' in X_out.columns:
            X_out['utilization_x_debtratio'] = (
                X_out['RevolvingUtilizationOfUnsecuredLines'] * X_out['DebtRatio']
            )
        
        # Ділимо місячний дохід на кількість утриманців (+1 у знаменнику проти ділення на нуль)
        if 'MonthlyIncome' in X_out.columns and 'NumberOfDependents' in X_out.columns:
            X_out['income_per_dependent'] = (
                X_out['MonthlyIncome'] / (X_out['NumberOfDependents'] + 1)
            )
        return X_out


class OutofFoldWoEEncoder(BaseEstimator, TransformerMixin):
    """
    Категорія В: Регуляризований Out-Of-Fold (OOF) Weight of Evidence (WoE) Encoder.
    Захищає від перенавчання на рідкісних категоріях.
    """
    def __init__(self, cat_cols=None, n_splits=5, smooth=10.0):
        self.cat_cols = cat_cols or []
        self.n_splits = n_splits
        self.smooth = smooth
        self.global_woe_ = {}

    # рахує формулу WoE для будь-якого шматочка даних
    def _calc_woe(self, df, col, target_col):
        total_pos = (df[target_col] == 1).sum()
        total_neg = (df[target_col] == 0).sum()
        
        grouped = df.groupby(col, observed=False)[target_col].agg(['count', 'sum'])
        pos = grouped['sum']
        neg = grouped['count'] - pos
        
        woe = np.log(((pos + self.smooth) / (total_pos + self.smooth)) / 
                     ((neg + self.smooth) / (total_neg + self.smooth)))
        return woe.to_dict()

    # Функція Запам'ятовування: рахує WoE глобально для всього датасету (для майбутніх нових заявок)
    def fit(self, X, y):
        X_df = pd.DataFrame(X).copy()
        X_df['target'] = np.array(y)
        
        for col in self.cat_cols:
            if col in X_df.columns:
                self.global_woe_[col] = self._calc_woe(X_df, col, 'target')
        return self

    def fit_transform(self, X, y):
        X_df = pd.DataFrame(X).copy()
        X_df['target'] = np.array(y)

        # Створюємо ділитель на 5 шматочків
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        X_out = X_df.copy()
        
        for col in self.cat_cols:
            if col not in X_df.columns:
                continue
            
            oof_series = pd.Series(index=X_df.index, dtype=float)
            # Проходимося по кожному з 5 шматочків
            for train_idx, val_idx in kf.split(X_df):
                train_fold = X_df.iloc[train_idx]
                # 1. Рахуємо WoE ТІЛЬКИ на цих 80% даних
                fold_woe = self._calc_woe(train_fold, col, 'target')
                # 2. Застосовуємо ці числа до залишених 20% даних (val_idx)
                oof_series.iloc[val_idx] = X_df.iloc[val_idx][col].map(fold_woe)

            # Також обчислюємо глобальний словник для виклику transform() на тест-вибірці
            self.global_woe_[col] = self._calc_woe(X_df, col, 'target')
            global_mean_woe = np.mean(list(self.global_woe_[col].values())) if self.global_woe_[col] else 0.0
            # Записуємо отримані безпечні WoE-числа у нову колонку
            X_out[f'{col}_woe'] = oof_series.fillna(global_mean_woe)
            
        return X_out.drop(columns=['target'])

    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        for col in self.cat_cols:
            if col in X_out.columns:
                global_mean = np.mean(list(self.global_woe_[col].values())) if self.global_woe_[col] else 0.0
                X_out[f'{col}_woe'] = X_out[col].map(self.global_woe_[col]).fillna(global_mean)
        return X_out