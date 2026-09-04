
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the Tourism dataset directly from the repository data folder
df = pd.read_csv("tourism_project/data/tourism.csv", index_col=0)

# Remove unnecessary identifier column
# CustomerID does not provide useful predictive information
df.drop(columns=["CustomerID"], inplace=True)

# NOTE:
# Categorical columns are intentionally kept as raw strings.
# The training pipeline will handle categorical encoding using OneHotEncoder.
# This keeps the same representation during training and Streamlit prediction.

# Separate features and target
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split the data into training and testing sets
# stratify=y keeps the proportion of customers who purchased the package
# consistent in both training and testing datasets.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save the prepared datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data preparation completed successfully.")
print(f"Training features shape: {Xtrain.shape}")
print(f"Testing features shape: {Xtest.shape}")
print(f"Training target shape: {ytrain.shape}")
print(f"Testing target shape: {ytest.shape}")

print("\nProdTaken distribution in training data:")
print(ytrain.value_counts())

print("\nProdTaken distribution in testing data:")
print(ytest.value_counts())

print("\nCategorical values are kept as raw strings for pipeline encoding.")
