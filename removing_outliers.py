from utils.utils import load_config, iqr, percentiles
import pandas as pd
import numpy as np

# Loads config file
config = load_config()

# Loads primary data
df = pd.read_csv(config["paths"]["primary"])

print(f"Initial len: {len(df)}")
print(f"Inital nans: {df.isnull().sum()}")

# Performs basic cleaning
# Gets the minimum and maximum allowed from the config file
# and set values outside this range as null
for field, bounds in config["data_cleaning"]["basic_cleaning"].items():
    len_removed = len(
        df.loc[(~df[field].between(bounds[0], bounds[1])) & (~df[field].isnull())]
    )
    pct_removed = len_removed / len(df.loc[~df[field].isnull()])
    df.loc[~df[field].between(bounds[0], bounds[1]), field] = np.nan
    print(
        f"Basic cleaning: field: {field} removed {len_removed} observations ({pct_removed})"
    )

# Removing outliers
# For the fields defined on the config file, identify outliers using
# iqr or percentiles method and set as null values identified as outliers
for field in config["data_cleaning"]["outliers"]:
    for crawler in df["crawler"].unique():
        method = "iqr"
        lower_bound, upper_bound, pct_outliers = iqr(
            df.loc[df["crawler"] == crawler, field]
        )
        if pct_outliers > config["data_cleaning"]["percentiles_threshold"]:
            method = "percentile"
            lower_bound, upper_bound, pct_outliers = percentiles(
                df.loc[df["crawler"] == crawler, field]
            )
        print(
            f"Outliers: field: {field}, crawler: {crawler}, method: {method}, pct_outliers: {pct_outliers}"
        )
        df.loc[
            (df["crawler"] == crawler) & (~df[field].between(lower_bound, upper_bound)),
            field,
        ] = np.nan

# Drop null values
if config["data_cleaning"]["drop_na"]:
    df = df.dropna()

# Save data to CSV
print(f"Final len: {len(df)}")
print(f"Saving to {config['paths']['model_input']}")
df.to_csv(config["paths"]["model_input"], index=False)
