import pandas as pd
import numpy as np
from scipy.stats import zscore, mode
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

csv_path = "data/raw.csv"
df = pd.read_csv(csv_path)

# drop unnecessary columns
cols_to_drop = [
    "address_id", "individual_id", "latitude", "longitude",
    "date_of_birth", "social_security_number",
    "state", "cust_orig_date", "acct_suspd_date"
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# calculate premium-to-income ratio
df["premium_to_income_ratio"] = df["curr_ann_amt"] / df["income"]

# convert marital_status to binary
df["marital_status"] = df["marital_status"].map(lambda x: 1 if str(x).strip().lower() == "married" else 0)

# replace home_market_value ranges with midpoints of ranges
def home_value_midpoint(value):
    try:
        if pd.isnull(value):
            return np.nan
        value = str(value).replace(",", "").strip()
        if " - " in value:
            low, high = value.split(" - ")
            return (float(low) + float(high)) / 2
        else:
            return float(value)
    except:
        return np.nan

if "home_market_value" in df.columns:
    df["home_market_value"] = df["home_market_value"].apply(home_value_midpoint)
    # set to 0 for non-homeowners
    if "home_owner" in df.columns:
        df.loc[df["home_owner"] == 0, "home_market_value"] = 0

# knn imputation for missing home_market_value
def knn_mode_impute(df, target_col, feature_cols, n_neighbors=5):
    known = df[df[target_col].notnull()]
    missing = df[df[target_col].isnull()]
    if missing.empty:
        return df
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
    nbrs.fit(known[feature_cols].to_numpy())
    for idx, row in missing.iterrows():
        distances, indices = nbrs.kneighbors([row[feature_cols].to_numpy()])
        neighbor_values = known.iloc[indices[0]][target_col].to_numpy()
        df.at[idx, target_col] = mode(neighbor_values, keepdims=True).mode[0]
    return df

feature_cols_for_knn = ["curr_ann_amt", "days_tenure", "age_in_years", "income",
                        "length_of_residence", "premium_to_income_ratio"]
if "home_market_value" in df.columns:
    df = knn_mode_impute(df, "home_market_value", feature_cols_for_knn, n_neighbors=5)

# ensure binary columns are integers
binary_cols = ["good_credit", "college_degree", "home_owner", "marital_status", "has_children"]
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].astype(int)

# ensure churn is the last column
if "Churn" in df.columns:
    cols = [c for c in df.columns if c != "Churn"] + ["Churn"]
    df = df[cols]

# train val test split
train_val, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df["Churn"])
train, val = train_test_split(train_val, test_size=0.1111, random_state=42, stratify=train_val["Churn"])

# target encoding for city and county
def target_encode(train_df, val_df, test_df, col, target):
    means = train_df.groupby(col)[target].mean()
    global_mean = train_df[target].mean()
    # apply map, fallback to global mean for unseen categories
    train_df[col] = train_df[col].map(means).fillna(global_mean)
    val_df[col] = val_df[col].map(means).fillna(global_mean)
    test_df[col] = test_df[col].map(means).fillna(global_mean)
    return train_df, val_df, test_df

for cat_col in ["city", "county"]:
    if cat_col in train.columns:
        train, val, test = target_encode(train, val, test, cat_col, "Churn")

# z-score normalization for numeric columns
numeric_cols = ["curr_ann_amt", "days_tenure", "age_in_years", "income",
                "length_of_residence", "premium_to_income_ratio", "home_market_value"]

scaling_params = {}
for col in numeric_cols:
    if col in train.columns:
        mean = train[col].mean()
        std = train[col].std(ddof=0)
        scaling_params[col] = (mean, std)
        train[col] = (train[col] - mean) / std
        val[col] = (val[col] - mean) / std
        test[col] = (test[col] - mean) / std

# smote on train set
X_train = train.drop(columns=["Churn"])
y_train = train["Churn"]
X_val = val.drop(columns=["Churn"])
y_val = val["Churn"]
X_test = test.drop(columns=["Churn"])
y_test = test["Churn"]

smote = SMOTE(sampling_strategy=1.0, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

train_res = pd.concat([X_train_res, y_train_res], axis=1)

# save outputs
train_res.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)

print(f"TRAIN SHAPE (after SMOTE): {train_res.shape}")
print(f"VAL SHAPE: {val.shape}")
print(f"TEST SHAPE: {test.shape}")

print("\nCLASS DISTRIBUTION AFTER SMOTE:")
print(train_res["Churn"].value_counts())
