
export HF_ALLOW_CODE_EVAL=1
LOCAL_CKPT_PATH_BASE=checkpoints
mkdir -p "${LOCAL_CKPT_PATH_BASE}"

ckp="hf_iter_$1"
task="$2"

MODEL_BASE_PATH="/mnt/blob-pretraining-hptraining/long_corpus/checkpoints"
MODEL_NAME="lcft_Meta-Llama-3-8B_ready_book-odl"
CKPT="checkpoint-$1"
MODEL_PATH="${MODEL_BASE_PATH}/${MODEL_NAME}/${CKPT}"

# HUGGINGFACE_CKPT_PATH="${LOCAL_CKPT_PATH_BASE}/${ckp}"
echo "Evaluating checkpoint: ${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python3.12 run_vllm_eval.py \
    --tensor_parallel_size 8 \
    --hf_path "${MODEL_PATH}" \
    --tasks ${task} \
    --num_fewshot 0 \
    --log_samples \
    --output_path "${EVAL_FOLDER}/eval_${task}_${ckp}" 

azcopy copy --recursive "${EVAL_FOLDER}/eval_${task}_${ckp}" "https://hptrainingsouthcentralus.blob.core.windows.net/pretraining/checkpoints/${CHECKPOINT_NAME}/${EVAL_FOLDER}/${SAS_KEY}"

rm -rf "${EVAL_FOLDER}/eval_${task}_${ckp}"
rm -rf "${LOCAL_CKPT_PATH_BASE}/${ckp}"