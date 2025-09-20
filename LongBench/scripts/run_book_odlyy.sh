model_name="book-odlyy"
python pred.py --model $model_name
python eval.py --model $model_name

BLOB_PATH="/mnt/blob-pretraining-hptraining/haoran_result/LongBench"
mkdir -p $BLOB_PATH
mv pred/${model_name} ${BLOB_PATH}/${model_name}