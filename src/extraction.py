import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class TextExtractor:
    def __init__(self, max_features=500, n_components=5):
        # 1. TF-IDF: підраховує важливість слів у тексті
        self.tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
        
        # 2. TruncatedSVD: стискає величезну таблицю слів всього до 5 найважливіших числових компонент (n_components=5)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)

    def fit_transform(self, text_list):
        # Перевіряємо кожен рядок: якщо текст порожній, None або складається з пробілів, 
        # ми замінюємо його на дефолтне слово "none", щоб код не впав з помилкою!
        text_list = [str(t) if pd.notna(t) and str(t).strip() != '' else "none" for t in text_list]
        
        # Перетворюємо текст на матрицю частоти слів
        tfidf_mat = self.tfidf.fit_transform(text_list)
        
        # Стискаємо матрицю слів до 5 колонок чисел
        text_emb = self.svd.fit_transform(tfidf_mat)
        return pd.DataFrame(text_emb, columns=[f'text_svd_{i}' for i in range(text_emb.shape[1])])

    def transform(self, text_list):
        # Метод для ТЕСТОВОЇ вибірки: робить те саме перетворення,
        # але використовує вже навчений TF-IDF та SVD з fit_transform()
        text_list = [str(t) if pd.notna(t) and str(t).strip() != '' else "none" for t in text_list]
        tfidf_mat = self.tfidf.transform(text_list)
        text_emb = self.svd.transform(tfidf_mat)
        return pd.DataFrame(text_emb, columns=[f'text_svd_{i}' for i in range(text_emb.shape[1])])

class SignalExtractor:
    def fit_transform(self, signal_arrays):
        features = []
        for sig in signal_arrays:
            # Якщо список транзакцій порожній, дорівнює None або містить лише NaN,
            # ми повертаємо нулі [0.0, 0.0, 0.0], щоб не зупиняти обчислення
            if sig is None or len(sig) == 0 or np.isnan(sig).all():
                features.append([0.0, 0.0, 0.0])
                continue
            
            # Перетворюємо у формат NumPy масиву
            sig = np.array(sig, dtype=float)
            # Замінюємо випадкові неіснуючі числа (NaN/Inf) на звичайні нулі
            sig = np.nan_to_num(sig)
            
            # Обчислюємо 3 головні показники сигналу/транзакцій:
            mean_val = np.mean(sig) # Середній чек/витрата
            std_val = np.std(sig)   # Волатильність (коливання витрат)
            max_val = np.max(sig)   # Найбільша пикова транзакція
            
            features.append([mean_val, std_val, max_val])
            
        # Повертаємо датафрейм із 3 новими числово-сигнальними колонками
        return pd.DataFrame(features, columns=['signal_mean', 'signal_std', 'signal_max'])

    def transform(self, signal_arrays):
        # Оскільки тут немає параметрів, які треба навчити, transform просто викликає fit_transform
        return self.fit_transform(signal_arrays)