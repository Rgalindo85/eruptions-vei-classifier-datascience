import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


PROJECT_DIR = Path(__file__).resolve().parents[1]
print(PROJECT_DIR)


def main():
    df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "processed", "final_volcanic_data.csv"))
    print(df.head())

    features = [
        'latitude',
        'longitude',
        'elevation',
        'eruption_start_year',
    ]

    cat = [
        'eruption_category',
        'primary_volcano_type',
        'evidence_category',
        'tectonic_settings',
        'major_rock_1',
    ]

    df_class = pd.get_dummies(df[features + cat], drop_first=True)

    X = df_class
    y1 = df['vei_class']
    y2 = df['vei']

    pca = PCA(n_components=0.95)  # Retain 95% of variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reduced = pca.fit_transform(X_scaled)
    exp_var = pca.explained_variance_ratio_

    # save the reduced dataset and explained variance
    reduced_df = pd.DataFrame(X_reduced)
    reduced_df['vei_class'] = y1.values
    reduced_df['vei'] = y2.values

    reduced_df.to_csv(os.path.join(PROJECT_DIR, "data", "processed", "pca_reduced_volcanic_data.csv"), index=False)
    np.save(os.path.join(PROJECT_DIR, "data", "processed", "pca_explained_variance.npy"), exp_var)


if __name__ == "__main__":
    main()