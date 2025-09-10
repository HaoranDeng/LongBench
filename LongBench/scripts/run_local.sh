MODEL_PATH="llama2-7b-chat-4k"

python pred.py --model $MODEL_PATH

python eval.py --model $MODEL_PATH

BLOB_PATH="/mnt/blob-pretraining-hptraining/haoran_results/LongBench"
mv pred/${model_name} $BLOB_PATH/${model_name}