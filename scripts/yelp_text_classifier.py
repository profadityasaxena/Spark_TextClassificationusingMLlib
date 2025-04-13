from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Start Spark session
spark = SparkSession.builder.appName("YelpTextClassifier").getOrCreate()

# File paths (adjust if needed)
train_path = "file:///home/cu/data/yelp_review_train.csv"
test_path = "file:///home/cu/data/yelp_review_test.csv"
output_path = "/home/cu/outputs/accuracy_results.csv"

# Load data
train_df = spark.read.option("header", True).csv(train_path)
test_df = spark.read.option("header", True).csv(test_path)

# Preprocessing pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label")

# Classifier
lr = LogisticRegression(featuresCol="tfidf_features", labelCol="indexed_label", maxIter=100)

# Build and train pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, label_indexer, lr])
model = pipeline.fit(train_df)

# Evaluate
evaluator = MulticlassClassificationEvaluator(labelCol="indexed_label", predictionCol="prediction", metricName="accuracy")

# Training accuracy
train_predictions = model.transform(train_df)
train_accuracy = evaluator.evaluate(train_predictions)

# Test accuracy
test_predictions = model.transform(test_df)
test_accuracy = evaluator.evaluate(test_predictions)

# Print results
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Save to CSV
results = spark.createDataFrame([(train_accuracy, test_accuracy)], ["Training Accuracy", "Test Accuracy"])
results.coalesce(1).write.mode("overwrite").option("header", True).csv(output_path)

# Done
spark.stop()
