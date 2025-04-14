from pyspark.ml.evaluation import MulticlassClassificationEvaluator

model = pipeline.fit(train_df)

train_predictions = model.transform(train_df)
test_predictions = model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="accuracy")
train_accuracy = evaluator.evaluate(train_predictions)
test_accuracy = evaluator.evaluate(test_predictions)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)