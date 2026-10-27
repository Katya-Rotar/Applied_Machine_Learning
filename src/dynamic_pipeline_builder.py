import numpy as np
import pandas as pd
import yaml
from typing import Dict, Any, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer


def validate_config(config: Dict[str, Any]) -> None:
    """
    Quality Gate 1: Перевірка конфігурації.
    Гарантує, що жодна захищена ознака (protected_attributes) не потрапить
    у список ознак для навчання (features).

    Пошук виконується до створення ColumnTransformer та завантаження даних.
    """
    protected = set(config.get("protected_attributes", []))
    features = set()
    
    # Збираємо всі ознаки з усіх груп features
    for group in config.get("features", {}).values():
        if isinstance(group, list):
            features.update(group)
            
    # Знаходимо перетин (витік)
    leaked = protected & features
    if leaked:
        raise ValueError(
            f"[QUALITY GATE ERROR] Захищені ознаки потрапили у features: {leaked}. "
            f"Необхідно вилучити їх з секції features у YAML конфігурації!"
        )


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Кастомний трансформер для логарифмування log1p(x) = log(1 + x)."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_curr = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        # Додаємо np.clip для уникнення від'ємних значень під логарифмом
        return np.log1p(np.clip(X_curr.values, a_min=0, a_max=None))


class WinsorizerTransformer(BaseEstimator, TransformerMixin):
    """Кастомний трансформер для обрізання викидів"""
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.lower_bounds_ = X_df.quantile(self.lower_quantile).values
        self.upper_bounds_ = X_df.quantile(self.upper_quantile).values
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        X_clipped = X_df.copy()
        for i in range(X_df.shape[1]):
            X_clipped.iloc[:, i] = np.clip(
                X_clipped.iloc[:, i], self.lower_bounds_[i], self.upper_bounds_[i]
            )
        return X_clipped.values


class AdaptiveImputer(BaseEstimator, TransformerMixin):
    """
    Адаптивний імпутер.
    Розраховує частку пропусків ТІЛЬКИ в методі fit() на X_train.
    Зберігає аудиторський слід (audit_trail_) для кожної колонки.
    """
    def __init__(self, policy: List[Dict[str, Any]], column_names: List[str]):
        self.policy = policy
        self.column_names = column_names
        self.audit_trail_ = {}
        self.transformers_ = []

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X, columns=self.column_names) if not isinstance(X, pd.DataFrame) else X.copy()
        self.audit_trail_ = {}
        self.transformers_ = []

        for col in self.column_names:
            # Обчислення частки пропусків ЛОКАЛЬНО у fit()
            missing_share = float(X_df[col].isna().mean())
            
            # Пошук відповідного правила з imputation_policy
            selected_rule = None
            for rule in sorted(self.policy, key=lambda r: r["max_share"]):
                if missing_share <= rule["max_share"]:
                    selected_rule = rule
                    break
            
            if selected_rule is None:
                selected_rule = self.policy[-1]

            # Запис в аудиторський слід
            self.audit_trail_[col] = {
                "missing_share": missing_share,
                "applied_rule": selected_rule
            }

            # Створення відповідного імпутера scikit-learn
            strat = selected_rule["strategy"]
            fill_val = selected_rule.get("fill_value", None)
            
            if strat == "median":
                imp = SimpleImputer(strategy="median")
            elif strat == "most_frequent":
                imp = SimpleImputer(strategy="most_frequent")
            elif strat == "constant":
                imp = SimpleImputer(strategy="constant", fill_value=fill_val)
            else:
                imp = SimpleImputer(strategy="median")

            imp.fit(X_df[[col]])
            self.transformers_.append((col, imp, selected_rule.get("add_indicator", False)))

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.column_names) if not isinstance(X, pd.DataFrame) else X.copy()
        transformed_cols = []

        for col, imp, add_ind in self.transformers_:
            col_data = X_df[[col]]
            
            # Створення індикатора пропуску, якщо вимагає політика
            if add_ind:
                indicator = col_data.isna().astype(float).values
                indicator_df = pd.DataFrame(indicator, columns=[f"{col}_isna"], index=X_df.index)
            
            imputed_array = imp.transform(col_data)
            imputed_df = pd.DataFrame(imputed_array, columns=[col], index=X_df.index)

            transformed_cols.append(imputed_df)
            if add_ind:
                transformed_cols.append(indicator_df)

        return pd.concat(transformed_cols, axis=1).values


class DynamicPipelineBuilder:
    """
    Конструктор конвеєра препроцесингу.
    Зчитує конфігурацію, проводить валідацію та повертає ненавчений ColumnTransformer.
    НЕ ЗВЕРТАЄТЬСЯ ДО ДАНИХ під час виконання build().
    """
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        # Виклик Quality Gate до збирання
        validate_config(self.config)

    def build(self) -> ColumnTransformer:
        """Збирає ненавчений ColumnTransformer на основі обраної model_family."""
        model_family = self.config["model_family"]
        transform_rules = self.config["transform_policy"][model_family]
        imputation_rules = self.config["imputation_policy"]

        transformers_list = []

        # 1. Гілка numeric_continuous
        num_cols = self.config["features"].get("numeric_continuous", [])
        if num_cols:
            num_steps = [
                ("adaptive_imputer", AdaptiveImputer(imputation_rules["numeric"], num_cols))
            ]
            
            methods = transform_rules.get("numeric_continuous", [])
            for method in methods:
                if method == "log1p":
                    num_steps.append(("log1p", Log1pTransformer()))
                elif method == "standard_scaler":
                    num_steps.append(("scaler", StandardScaler()))
                elif method == "winsorize":
                    num_steps.append(("winsorize", WinsorizerTransformer()))

            num_pipeline = Pipeline(num_steps)
            transformers_list.append(("numeric_continuous", num_pipeline, num_cols))

        # 2. Гілка categorical_nominal
        cat_nom_cols = self.config["features"].get("categorical_nominal", [])
        if cat_nom_cols:
            cat_nom_steps = [
                ("adaptive_imputer", AdaptiveImputer(imputation_rules["categorical"], cat_nom_cols))
            ]
            methods = transform_rules.get("categorical_nominal", [])
            for method in methods:
                if method == "one_hot":
                    cat_nom_steps.append(("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
                elif method == "woe":
                    # Спрощена імітація WoE через OneHot для демонстрації
                    cat_nom_steps.append(("one_hot_woe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))

            cat_nom_pipeline = Pipeline(cat_nom_steps)
            transformers_list.append(("categorical_nominal", cat_nom_pipeline, cat_nom_cols))

        # 3. Гілка categorical_ordinal
        cat_ord_cols = self.config["features"].get("categorical_ordinal", [])
        if cat_ord_cols:
            cat_ord_steps = [
                ("adaptive_imputer", AdaptiveImputer(imputation_rules["categorical"], cat_ord_cols))
            ]
            methods = transform_rules.get("categorical_ordinal", [])
            for method in methods:
                if method == "ordinal_encoder":
                    cat_ord_steps.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
                elif method == "standard_scaler":
                    cat_ord_steps.append(("scaler", StandardScaler()))

            cat_ord_pipeline = Pipeline(cat_ord_steps)
            transformers_list.append(("categorical_ordinal", cat_ord_pipeline, cat_ord_cols))

        # Збирання підсумкового ColumnTransformer
        column_transformer = ColumnTransformer(
            transformers=transformers_list,
            remainder="drop" # Гарантія виключення неназваних колонок (включаючи protected)
        )

        return column_transformer