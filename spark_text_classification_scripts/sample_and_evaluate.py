train_10k = train_df.sample(False, 10000 / train_df.count(), seed=42)
train_15k = train_df.sample(False, 15000 / train_df.count(), seed=42)
train_20k = train_df.sample(False, 20000 / train_df.count(), seed=42)
train_25k = train_df.sample(False, 25000 / train_df.count(), seed=42)
train_30k = train_df.sample(False, 30000 / train_df.count(), seed=42)

model_10k = pipeline.fit(train_10k)
acc_10k = evaluator.evaluate(model_10k.transform(test_df))
print("Test Accuracy @ 10K:", acc_10k)

model_15k = pipeline.fit(train_15k)
acc_15k = evaluator.evaluate(model_15k.transform(test_df))
print("Test Accuracy @ 15K:", acc_15k)

model_20k = pipeline.fit(train_20k)
acc_20k = evaluator.evaluate(model_20k.transform(test_df))
print("Test Accuracy @ 20K:", acc_20k)

model_25k = pipeline.fit(train_25k)
acc_25k = evaluator.evaluate(model_25k.transform(test_df))
print("Test Accuracy @ 25K:", acc_25k)

model_30k = pipeline.fit(train_30k)
acc_30k = evaluator.evaluate(model_30k.transform(test_df))
print("Test Accuracy @ 30K:", acc_30k)