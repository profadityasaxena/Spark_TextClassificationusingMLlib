import pandas as pd

results = {
    "Training Size": [10000, 15000, 20000, 25000, 30000, "Full (LR)", "Full (NB)"],
    "Model": ["Logistic Regression"]*5 + ["Logistic Regression", "Naive Bayes"],
    "Test Accuracy": [acc_10k, acc_15k, acc_20k, acc_25k, acc_30k, test_accuracy, nb_test_accuracy]
}

df_results = pd.DataFrame(results)
df_results.to_csv("/home/cu/outputs/accuracy_results.csv", index=False)
print(df_results)