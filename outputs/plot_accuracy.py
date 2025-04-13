import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy results
df = pd.read_csv("C:/Users/CODER/Desktop/Projects/Spark_TextClassification/Spark_TextClassification/outputs/accuracy_results.csv")

# Filter only rows with numeric training sizes
df_numeric = df[df["Training Size"].apply(lambda x: str(x).isdigit())]
df_numeric["Training Size"] = df_numeric["Training Size"].astype(int)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(df_numeric["Training Size"], df_numeric["Test Accuracy"], marker='o', label="Logistic Regression")

# Overlay Naive Bayes test accuracy
nb_accuracy = df[df["Model"] == "Naive Bayes"]["Test Accuracy"].values[0]
plt.axhline(y=nb_accuracy, color='orange', linestyle='--', label="Naive Bayes (Full Data)")

# Labels and layout
plt.title("Test Accuracy vs Training Size")
plt.xlabel("Training Size")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig("C:/Users/CODER/Desktop/Projects/Spark_TextClassification/Spark_TextClassification/outputs/accuracy_plot.png")
plt.show()
