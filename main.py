import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


def log(msg):
    print(msg, file=sys.stderr)


def main():
    df = pd.read_csv("loan_old.csv")
    df.drop(columns=["Loan_ID"], inplace=True)

    empty = df.isnull().sum().sum()
    log("There are " + str(empty) + " empty values, dropping rows containing them.")
    df.dropna(inplace=True)

    features_df = df.drop(columns=["Max_Loan_Amount", "Loan_Status"])
    targets_df = df[["Max_Loan_Amount", "Loan_Status"]]

    categorical_features_df = features_df.select_dtypes(include=["object"])
    numerical_features_df = features_df.select_dtypes(exclude=["object"])
    log("Categorical features:")
    for col in categorical_features_df.columns:
        log(f"\t- {col}")
    log("Numerical features:")
    for col in numerical_features_df.columns:
        log(
            f"\t- {col} ({numerical_features_df[col].min()} - {numerical_features_df[col].max()})"
        )

    test_size = 0.2
    train_size = 1 - test_size
    log(
        f"Shuffling and splitting the data {test_size * 100}% test data and {(train_size) * 100}% training data"
    )
    (
        features_train,
        features_test,
        max_loan_train,
        max_loan_test,
        loan_status_train,
        loan_status_test,
    ) = train_test_split(
        features_df,
        targets_df["Max_Loan_Amount"],
        targets_df["Loan_Status"],
        test_size=test_size,
        train_size=train_size,
        random_state=30,
    )

    label_encoders = {}
    standard_scalers = {}

    for col in features_train.columns:
        if features_train[col].dtype == "object":
            log(f"Encoding {col}")
            label_encoders[col] = LabelEncoder()
            features_train[col] = label_encoders[col].fit_transform(features_train[col])
            log(f"\t- Before: {label_encoders[col].classes_}")
            log(f"\t- After: {np.unique(features_train[col])}")
        else:
            log(f"Standardizing {col}")
            log(
                f"\t- Before: {np.min(features_train[col])} to {np.max(features_train[col])}"
            )
            standard_scalers[col] = StandardScaler()
            features_train[col] = standard_scalers[col].fit_transform(
                features_train[[col]]
            )
            log(
                f"\t- After: {np.min(features_train[col])} to {np.max(features_train[col])}"
            )

    log(f"Encoding Loan_Status")
    loan_status_encoder = LabelEncoder()
    loan_status_train = pd.Series(
        loan_status_encoder.fit_transform(loan_status_train),
        name=loan_status_train.name,
    )
    log(f"\t- Before: {loan_status_encoder.classes_}")
    log(f"\t- After: {np.unique(loan_status_train)}")

    sns.pairplot(numerical_features_df)
    plt.show()


if __name__ == "__main__":
    main()
