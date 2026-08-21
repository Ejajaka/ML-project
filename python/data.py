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


def load_actg175():
    """Load the ACTG175 HIV clinical trial (Hammer et al. 1996).

    This is the canonical real-RCT dataset used by the survival-RuleFit / SCRE
    papers (Wan et al. 2022, 2023, 2024; Hiraishi et al. 2023) to illustrate
    interpretable HTE, so it is the ideal verification dataset for CISCaRL:
    treatment is randomized (known propensity), outcomes are right-censored
    time-to-(AIDS or death), and published rule lists are available to compare
    against.

    Source: speff2trial R package / MIT HIV trial ACTG175 (local copy in
    data/ACTG175.csv; falls back to Rdatasets URL if present).
    Covariates: baseline age, weight, CD4/CD8, Karnofsky score, hemophilia,
    homosexual activity, IV drug use, prior zidovudine use, race, gender,
    symptomatic status.
    Treatment: treat == 1 is combination therapy (ddI or ddI+ZDV or ddC+ZDV),
    treat == 0 is ZDV monotherapy.
    Outcome: days until first CD4 decline>=50, AIDS progression, or death;
    cens == 1 indicates the event was observed.
    """
    import os
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'ACTG175.csv')
    if os.path.exists(local):
        df = pd.read_csv(local)
    else:
        url = ("https://raw.githubusercontent.com/vincentarelbundock/"
               "Rdatasets/master/csv/survival/ACTG175.csv")
        df = pd.read_csv(url)

    df["treatment"] = df["treat"].astype(int)
    df["time_days"] = df["days"]
    df["event"] = df["cens"].astype(int)

    # Baseline-only covariates (drop post-randomization measurements cd420,
    # cd496, cd820, r, offtrt, and outcomes/ids).
    exclude = ["pidnum", "treat", "days", "cens", "treatment", "event",
               "time_days", "cd420", "cd496", "r", "cd820", "offtrt"]
    covariates = [c for c in df.columns if c not in exclude]

    # Encode categoricals and impute with median
    for col in covariates:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = pd.factorize(df[col])[0]
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df, covariates


if __name__ == "__main__":
    df, covs = load_pbc()
    print(f"Loaded PBC: {len(df)} patients, {len(covs)} covariates")
    print(f"Treatment: {df['treatment'].sum()}, Control: {(1-df['treatment']).sum()}")
    print(f"Events: {df['event'].sum()}, Censored: {(1-df['event']).sum()}")
    print(f"Covariates: {covs}")

    df2, covs2 = load_actg175()
    print(f"\nLoaded ACTG175: {len(df2)} patients, {len(covs2)} covariates")
    print(f"Treatment: {df2['treatment'].sum()}, Control: {(1-df2['treatment']).sum()}")
    print(f"Events: {df2['event'].sum()}, Censored: {(1-df2['event']).sum()}")
    print(f"Covariates: {covs2}")
