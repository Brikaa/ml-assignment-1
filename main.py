import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def log(msg):
    print(msg, file=sys.stderr)


def main():
    df = pd.read_csv("loan_old.csv")
    df.drop(columns=["Loan_ID"], inplace=True)

    empty = df.isnull().sum().sum()
    log("There are " + str(empty) + " empty values, dropping them.")
    df.dropna(inplace=True)

    features_df = df.drop(columns=["Max_Loan_Amount", "Loan_Status"])
    targets_df = df[["Max_Loan_Amount", "Loan_Status"]]

    log("Types of the features:")
    for col in features_df.columns:
        t = "categorical" if features_df[col].dtype == "object" else "numerical"
        log(f"\t- {col} is {t}")
        if t == "numerical":
            log("\t\tScale:")
            log(f"\t\t- max: {features_df[col].max()}" )
            log(f"\t\t- min: {features_df[col].min()}" )
            log(f"\t\t- mean: {features_df[col].mean()}" )

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



if __name__ == "__main__":
    main()
