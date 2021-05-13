import pandas as pd
from utils.utils import load_config

# Loads configuration
config = load_config()

# Loads raw data
df = pd.read_csv(config["paths"]["raw"])
print(f"Initial len: {len(df)}")

# drop duplicates
len_before = len(df)
df = df.drop_duplicates(
    subset=config["data_cleaning"]["duplicates_subset"], keep="first"
).reset_index(drop=True)
len_after = len(df)
print(f"Removed {len_before-len_after} due to duplicates")

# drop ads
len_before = len(df)
ads = [_id.isnumeric() for _id in df[config["data_cleaning"]["ads_id"]]]
df = df[ads].reset_index(drop=True)
len_after = len(df)
print(f"Removed {len_before-len_after} due to ads")

# Transform text columns to int
for field in config["data_cleaning"]["text_to_int"]:
    print(f"Converting {field} from text to int")
    df[field] = df[field].str.split(" ").str[0].str.replace("--", "0").astype(int)

# Transform numeric saved as str to int
for field in config["data_cleaning"]["numeric_to_int"]:
    print(f"Converting {field} from numeric to int")
    df[field] = df[field].astype(int)

# Transform money to float
for field in config["data_cleaning"]["money_to_float"]:
    print(f"Converting {field} from money to float")
    df[field] = (
        df[field]
        .str.split("$")
        .str[1]
        .str.split("/")
        .str[0]
        .str.replace(".", "", regex=True)
        .astype("float")
    )

# Transforms string to datetime
for field in config["data_cleaning"]["datetime"]:
    print(f"Converting {field} to datetime")
    df[field] = pd.to_datetime(df[field], format="%Y-%m-%d %H:%M")

# Normalize string field
for field in config["data_cleaning"]["str_to_normalize"]:
    print(f"Normalizing text field {field}")
    df[field] = (
        df[field]
        .str.replace(" ", "_")
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf8")
    )

# Transform amenities to dummies
for field in config["data_cleaning"]["amenities_to_dummies"]:
    print(f"Getting dummies from {field}")
    dummies = df[field].str.get_dummies("\n")
    dummies.columns = [
        w.lower().replace(" ", "_").encode("ascii", errors="ignore").decode("utf8")
        for w in dummies.columns
    ]
    df = pd.concat((df, dummies), axis=1)
    df = df.drop(columns=field)

# Saves to csv
print(f"Final len: {len(df)}")
print(f"Saving to {config['paths']['primary']}")
df.to_csv(config["paths"]["primary"], index=False)
