from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes(featuresCol="tfidf_features", labelCol="indexed_label")
nb_pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, label_indexer, nb])

nb_model = nb_pipeline.fit(train_df)
nb_test_preds = nb_model.transform(test_df)
nb_test_accuracy = evaluator.evaluate(nb_test_preds)
print("Naive Bayes Test Accuracy:", nb_test_accuracy)