Instructions to generate the data for workloads 1-4 from [BigVectorBench](https://github.com/BenchCouncil/BigVectorBench).
Data can be loaded from https://huggingface.co/datasets/Patrickcode/BigVectorBench/tree/main
Data can be converted to parquet with commands:

```
python3 generate_parquet.py ag_news-384-euclidean-filter.hdf5 parquet/ag_news_test.parquet test_vec test_label neighbors distances
python3 generate_parquet.py ag_news-384-euclidean-filter.hdf5 parquet/ag_news_train.parquet train_vec train_label

python3 generate_parquet.py cc_news-384-euclidean-filter.hdf5 parquet/cc_news_test.parquet test_vec test_label neighbors distances
python3 generate_parquet.py cc_news-384-euclidean-filter.hdf5 parquet/cc_news_train.parquet train_vec train_label

python3 generate_parquet.py app_reviews-384-euclidean-filter.hdf5 parquet/app_reviews_test.parquet test_vec test_label neighbors distances
python3 generate_parquet.py app_reviews-384-euclidean-filter.hdf5 parquet/app_reviews_train.parquet train_vec train_label

python3 generate_parquet_chunked.py amazon-384-euclidean-5filter.hdf5 parquet/amazon_test.parquet test_vec test_label neighbors distances
python3 generate_parquet_chunked.py amazon-384-euclidean-5filter.hdf5 parquet/amazon_train.parquet train_vec train_label

python3 generate_parquet.py ag_news-384-euclidean.hdf5 parquet/ag_news_unfiltered_train.parquet train -v
python3 generate_parquet.py ag_news-384-euclidean.hdf5 parquet/ag_news_unfiltered_train.parquet test neighbors distances -v
```


