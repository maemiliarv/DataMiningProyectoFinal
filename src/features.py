import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a set of generic feature engineering steps:
      - Parse datetime columns and extract components
      - Impute numeric NaNs with median
      - One-hot encode categorical features
      - Create custom ratio features
    """
    df = df.copy()

    # 1) Datetime features
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                dt = pd.to_datetime(df[col], errors='raise')
                df[col + '_year'] = dt.dt.year
                df[col + '_month'] = dt.dt.month
                df[col + '_day'] = dt.dt.day
                df[col + '_weekday'] = dt.dt.weekday
            except Exception:
                pass

    # 2) Numeric imputation
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # 3) Categorical encoding
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 4) Custom ratio features (example)
    if 'revenue' in df.columns and 'costs' in df.columns:
        df['revenue_cost_ratio'] = df['revenue'] / (df['costs'] + 1e-6)

    return df


def load_and_process(path: str) -> pd.DataFrame:
    """
    Load raw CSV and apply feature engineering.
    """
    df = pd.read_csv(path)
    return create_features(df)