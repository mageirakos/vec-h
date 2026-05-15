# maxvs - instructions to generate dataset

By following the steps described in this project one can recreate the dataset.
The dataset can be generated in both *parquet* as well as *csv* format.
The vector/embedding columns are only available in parquet and are stored as parquet lists.

For Vec-H SF 1 used in the paper : https://polybox.ethz.ch/index.php/s/tcFK2AHaMaBKHyw

## Steps
If you dont' have uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

1. **Set up project**
```shell
uv sync
source .venv/bin/activate
mkdir data
```

2. **Generate TPCH data**
For a sample dataset, set e.g. `--sf 0.001 --n_files 1`. When using a small scale factor, also pass `--n_query_samples` accordingly in step 5.
```shell
python generate_tpch_tables.py --sf 1 --n_files 1 --output_dir_parquet data/tpch-sf1/parquet --output_dir_csv data/tpch-sf1/csv
```

3. **Embeddings data downloading**
```shell
python download_amazon23.py data/amazon-23/raw --parallel 2 --categories Industrial_and_Scientific
```

4. **Embeddings computation**
Appropriate hardware is recommended to run this step efficiently. 
Can be long-running, recommended to use tmux/screen.

```shell
# Review text embeddings (Qwen 0.6B)
python generate_embeddings_review.py \
    --input-dir data/amazon-23/raw/reviews \
    --output-dir data/amazon-23/parquet/Qwen0.6B/reviews \
    --batch-size 8

# Product image embeddings (SigLIP2)
python generate_embeddings_img.py \
    --input-dir data/amazon-23/raw/meta \
    --output-dir data/amazon-23/parquet/google_siglip2-so400m-patch14-384/images \
    --batch-size 4
```
If you downloaded multiple categories, and you only want to create embeddings for one, then you should have a per-category subfolder structure.

5. **Map embeddings to TPC-H data**
```shell
python generate_vech_tables.py \
    --input_dir_tpch_parquet data/tpch-sf1/parquet \
    --input_dir_reviews_parquet data/amazon-23/parquet/Qwen0.6B/reviews \
    --input_dir_imgs_parquet data/amazon-23/parquet/google_siglip2-so400m-patch14-384/images \
    --output_dir_parquet data/vech/parquet \
    --output_dir_csv data/vech 2>&1 | tee data/vech.log
```

6. **Copy Final Dataset to target path**
Run from `dataset-generation/`
```shell
mkdir ../../data/vech-sf1
cp -r data/vech/parquet/* data/tpch-sf1/csv/* "../../data/vech-sf1"/
cp -r data/tpch-sf1/parquet "../../data/vech-sf1/tpch_parquet-sf1"
```