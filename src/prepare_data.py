import os
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

def main():
    print("****** Starting data preparation ******")
    df = get_data()

    df_train, df_test = split_data(df)

    print(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    # save processed data
    save_data(df_train, "train.csv")
    save_data(df_test, "test.csv")

    print("DONE!!")


def save_data(df, filename):
    filepath = os.path.join(PROJECT_DIR, "data", "model_input")
    os.makedirs(filepath, exist_ok=True)

    df.to_csv(os.path.join(filepath, filename), index=False)

def split_data(df, train_frac=0.7):
    # shuffling the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # split into train and test
    train_size = int(len(df) * train_frac)

    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]
    return df_train, df_test


def get_data():
    df_volcano, df_eruptions, df_events = get_volcano_data()

    # Merge datasets
    df = df_eruptions.merge(df_volcano, on="volcano_number", how="left")
    df = df.merge(df_events, on="eruption_number", how="left")
    
    # Clean up memory
    del df_volcano, df_eruptions, df_events
   
    # Prepare final dataset
    df = prepare_final_dataset(df)
    print(f"Final dataset shape: {df.shape}")
    return df


def prepare_final_dataset(df):
    df["vei"] = pd.to_numeric(df["vei"], errors="coerce")
    df = df.dropna(subset=["vei"])

    df["vei_class"] = df["vei"].apply(lambda x: "low" if x <= 2 else "high")
    df["vei_class_label"] = df["vei_class"].map({"low":0, "high":1})

    return df

def get_volcano_data():
    df_volcano = pd.read_csv(os.path.join(PROJECT_DIR, "data", "clean", "volcano.csv"))
    df_eruptions = pd.read_csv(os.path.join(PROJECT_DIR, "data", "clean", "eruptions.csv"))
    df_events = pd.read_csv(os.path.join(PROJECT_DIR, "data", "clean", "events.csv"))
    
    return df_volcano, df_eruptions, df_events

if __name__ == "__main__":
    main()