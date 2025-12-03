import pandas as pd
import os

CSV = "features_test.csv"
CLEAN_CSV = "features_test.clean.csv"

if not os.path.exists(CSV):
    raise FileNotFoundError(f"Input CSV not found: {CSV}")

df = pd.read_csv(CSV, dtype=str)
if "label" not in df.columns:
    raise ValueError(f"El CSV {CSV} no contiene la columna 'label'. Columnas: {list(df.columns)}")

df['label'] = df['label'].astype(str).str.strip()
feat_cols = [c for c in df.columns if c != 'label']

# Convertir features a numérico (coerce errores a NaN)
df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors='coerce')

initial_len = len(df)
# Eliminar filas con NaN en features
df = df.dropna(subset=feat_cols)
# Eliminar labels vacíos
df = df[df['label'].str.strip() != ""]
# Eliminar labels que contengan solo '=' repetidos (por ejemplo '=======')
df = df[~df['label'].str.contains('=+')]

dropped = initial_len - len(df)

df.to_csv(CLEAN_CSV, index=False, encoding='utf-8')
print(f"Saved cleaned CSV to {CLEAN_CSV}. Dropped {dropped} invalid rows. Remaining: {len(df)}")
