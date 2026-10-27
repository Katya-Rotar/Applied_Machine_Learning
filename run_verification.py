import yaml
from src.dynamic_pipeline_builder import DynamicPipelineBuilder

def test_idempotent(config_path: str):
    """Два послідовні збирання з тією самою конфігурацією дають ідентичну структуру."""
    a = DynamicPipelineBuilder(config_path).build()
    b = DynamicPipelineBuilder(config_path).build()
    assert repr(a) == repr(b), f"Збирання не ідемпотентне для {config_path}"

def test_protected_excluded(config_path: str):
    """Жодна захищена ознака не потрапила до жодної гілки конвеєра."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    ct = DynamicPipelineBuilder(config_path).build()
    
    # Збираємо всі колонки, які явно використовуються трансформерами
    used_cols = set()
    for name, trans, cols in ct.transformers:
        if isinstance(cols, list):
            used_cols.update(cols)
        elif isinstance(cols, str):
            used_cols.add(cols)
            
    protected_set = set(cfg.get("protected_attributes", []))
    leaked = used_cols & protected_set
    assert not leaked, f"Захищені ознаки виявлені у конвеєрі: {leaked}"

if __name__ == "__main__":
    for path in ("config/config_lr.yaml", "config/config_rf.yaml"):
        test_idempotent(path)
        test_protected_excluded(path)
        print(f"[{path}]: ВСІ ПЕРЕВІРКИ УСПІШНО ПРОЙДЕНО!")