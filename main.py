import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


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

    categorical_df = features_df.select_dtypes(include=["object"])
    numerical_df = features_df.select_dtypes(include=["int64", "float64"])
    log("Categorical features:")
    for col in categorical_df.columns:
        log(f"\t- {col}")
    log("Numerical features:")
    for col in numerical_df.columns:
        log(f"\t- {col} ({numerical_df[col].min()} - {numerical_df[col].max()})")

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

    sns.pairplot(numerical_df)
    plt.show()


if __name__ == "__main__":
    main()
