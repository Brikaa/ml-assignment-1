# %% [markdown]
# # Omar Adel Abdel Hamid Ahmed Brikaa - 20206043 - S5

# %% [markdown]
# # Imports

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from IPython.display import display

# %% [markdown]
# # Check whether there are missing values

# %%
df = pd.read_csv("loan_old.csv")
empty = df.isnull().sum().sum()
print("There are " + str(empty) + " empty values")

# %% [markdown]
# # Records containing missing values are removed

# %%
clean_df = df.drop(columns=["Loan_ID"]).dropna()
display(clean_df)

# %% [markdown]
# # Check the type of each feature, and the scale of numerical features (implies separating the features and the targets)

# %%
features_df = clean_df.drop(columns=["Max_Loan_Amount", "Loan_Status"])
targets_df = clean_df[["Max_Loan_Amount", "Loan_Status"]]

categorical_features_df = features_df.select_dtypes(include=["object"])
numerical_features_df = features_df.select_dtypes(exclude=["object"])

print("Categorical features:")
for col in categorical_features_df.columns:
    print(f"\t- {col}")
print("Numerical features:")
for col in numerical_features_df.columns:
    print(
        f"\t- {col} ({numerical_features_df[col].min()} - {numerical_features_df[col].max()})"
    )

# %% [markdown]
# # Visualize a pairplot between numerical columns

# %%
sns.pairplot(clean_df.select_dtypes(exclude=["object"]))
plt.show()

# %% [markdown]
# # The data is shuffled and split into training and testing sets

# %%
test_size = 0.2
train_size = 1 - test_size

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

print("Features training set")
display(features_train)
print("Features testing set")
display(features_test)
print("Max loan (target) training")
display(pd.DataFrame(max_loan_train))
print("Max loan (target) testing")
display(pd.DataFrame(max_loan_test))
print("Loan status (target) training")
display(pd.DataFrame(loan_status_train))
print("Loan status (target) testing")
display(pd.DataFrame(loan_status_test))

# %% [markdown]
# # Encode and standardize training data

# %%
label_encoders = {}
standard_scalers = {}
processed_features_train = pd.DataFrame(index=features_train.index)

for col in features_train.columns:
    if features_train[col].dtype == "object":
        print(f"Encoding {col}")
        label_encoders[col] = LabelEncoder()
        processed_features_train[col] = label_encoders[col].fit_transform(features_train[col])
        print(f"\t- Before: {label_encoders[col].classes_}")
        print(f"\t- After: {np.unique(processed_features_train[col])}")
    else:
        print(f"Standardizing {col}")
        print(
            f"\t- Before: {np.min(features_train[col])} to {np.max(features_train[col])}"
        )
        standard_scalers[col] = StandardScaler()
        processed_features_train[col] = standard_scalers[col].fit_transform(features_train[[col]])
        print(
            f"\t- After: {np.min(processed_features_train[col])} to {np.max(processed_features_train[col])}"
        )

print(f"Encoding training Loan_Status")
loan_status_encoder = LabelEncoder()
processed_loan_status_train = pd.Series(
    loan_status_encoder.fit_transform(loan_status_train),
    name=loan_status_train.name,
)
print(f"\t- Before: {loan_status_encoder.classes_}")
print(f"\t- After: {np.unique(processed_loan_status_train)}")


# %% [markdown]
# # Fit a linear regression model to the data to predict the loan amount

# %%
linear_model = LinearRegression()
linear_model.fit(processed_features_train, max_loan_train)
print(linear_model.feature_names_in_)
print(linear_model.coef_)

# %% [markdown]
# # Evaluate the linear regression model using sklearn's R2 score

# %%
def preprocess_new_features(label_encoders, standard_scalers, new_features_df):
    processed_new_features_df = pd.DataFrame(index=new_features_df.index)
    for col in new_features_df.columns:
        if new_features_df[col].dtype == "object":
            print(f"Encoding {col}")
            processed_new_features_df[col] = label_encoders[col].transform(
                new_features_df[col]
            )
            print(f"\t- Before: {label_encoders[col].classes_}")
            print(f"\t- After: {np.unique(processed_new_features_df[col])}")
        else:
            print(f"Standardizing {col}")
            print(
                f"\t- Before: {np.min(new_features_df[col])} to {np.max(new_features_df[col])}"
            )
            processed_new_features_df[col] = standard_scalers[col].transform(
                new_features_df[[col]]
            )
            print(
                f"\t- After: {np.min(processed_new_features_df[col])} to {np.max(processed_new_features_df[col])}"
            )
    return processed_new_features_df


processed_features_test = preprocess_new_features(
    label_encoders, standard_scalers, features_test
)
print(f"R^2 score: {linear_model.score(processed_features_test, max_loan_test)}")

# %% [markdown]
# # Fit a logistic regression model to the data to predict the loan status

# %%
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def lg_gradient_descent(learning_rate, epochs, X, y):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    m, n = X.shape
    weights = np.zeros(n)

    for _ in range(epochs):
        z = np.dot(X, weights)
        h = sigmoid(z)
        diff = h - y

        gradients = np.dot(X.T, diff) / m

        weights -= learning_rate * gradients

    return weights


def lg_predict(weights, X):
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    return (predictions > 0.5).astype(int)


X_lg_train = processed_features_train.values
y_lg_train = processed_loan_status_train.values
lg_learning_rate = 0.01
lg_epochs = 500
lg_weights = lg_gradient_descent(lg_learning_rate, lg_epochs, X_lg_train, y_lg_train)
print(lg_weights)

# %% [markdown]
# # Write a function (from scratch) to calculate the accuracy of the model

# %%
processed_loan_status_test = pd.Series(
    loan_status_encoder.transform(loan_status_test),
    name=loan_status_test.name,
)

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_pred == y_true)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

y_true = processed_loan_status_test.values
y_pred = lg_predict(lg_weights, processed_features_test)
print(f"Accuracy: {calculate_accuracy(y_true, y_pred) * 100:.2f}%")

# %% [markdown]
# # Load the "loan_new.csv" dataset, perform the same preprocessing on it (except shuffling and splitting)

# %%
new_df = pd.read_csv("loan_new.csv")
new_df_processed = new_df.dropna().copy()
new_features_processed = preprocess_new_features(
    label_encoders, standard_scalers, new_df_processed.drop(columns=["Loan_ID"])
)

# %% [markdown]
# # Use your models on this data to predict the loan amounts and status

# %%
new_maximum_loan_amounts = linear_model.predict(new_features_processed)
new_loan_statuses = lg_predict(lg_weights, new_features_processed)

new_df_processed["Max_Loan_Amount"] = new_maximum_loan_amounts
new_df_processed["Loan_Status"] = loan_status_encoder.inverse_transform(
    new_loan_statuses
)
with pd.option_context("display.max_rows", None):
    display(new_df_processed)


