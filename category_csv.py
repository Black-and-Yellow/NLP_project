import pandas as pd

# Load your dataset
df = pd.read_csv("dataset/data.csv", sep="|", names=["filename", "title", "extra", "category"])

# Display first few rows (optional)
print(df.head())

# Get unique categories
unique_categories = df["category"].unique()

# Count how many unique categories
num_categories = len(unique_categories)

print(f"Total unique categories: {num_categories}")
print("List of categories:")
for i, cat in enumerate(unique_categories, 1):
    print(f"{i}. {cat}")
