from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("LoadYelpData").getOrCreate()

train_df = spark.read.option("header", True).csv("file:///home/cu/data/yelp_review_train.csv")
test_df = spark.read.option("header", True).csv("file:///home/cu/data/yelp_review_test.csv")

train_df.printSchema()
train_df.show(5)
test_df.printSchema()
test_df.show(5)