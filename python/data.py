import pandas as pd
import numpy as np

def load_pbc():
    url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/pbc.csv"
    df = pd.read_csv(url)
    
    # Filter randomized patients only
    df = df[df["trt"].notna()].copy()
    df["trt"] = df["trt"].astype(int)
    
    # Treatment: 1 = D-penicillamine, 2 = placebo -> binary 1/0
    df["treatment"] = (df["trt"] == 1).astype(int)
    
    # Outcome: death (status=2) as event, censor transplants (status=1)
    df["event"] = (df["status"] == 2).astype(int)
    
    # Time is in days
    df["time_days"] = df["time"]
    
    # Select covariates (drop ids, derived, and outcome columns)
    exclude = ["rownames", "id", "time", "status", "trt", "treatment", "event", "time_days"]
    covariates = [c for c in df.columns if c not in exclude]
    
    # Encode categoricals
    for col in covariates:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]
    
    # Impute missing values with median
    for col in covariates:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df, covariates

if __name__ == "__main__":
    df, covs = load_pbc()
    print(f"Loaded PBC: {len(df)} patients, {len(covs)} covariates")
    print(f"Treatment: {df['treatment'].sum()}, Control: {(1-df['treatment']).sum()}")
    print(f"Events: {df['event'].sum()}, Censored: {(1-df['event']).sum()}")
    print(f"Covariates: {covs}")
