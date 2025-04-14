from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="text", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
label_indexer = StringIndexer(inputCol="label", outputCol="indexed_label")

lr = LogisticRegression(featuresCol="tfidf_features", labelCol="indexed_label", maxIter=100)

pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, label_indexer, lr])