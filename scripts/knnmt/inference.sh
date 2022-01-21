

export CUDA_VISIBLE_DEVICES=$1
DSTORE_SIZE=7000000
MODEL_PATH=/home/data_ti5_c/wangdq/model/fairseq/pretrain/wmt19/deen/model/wmt19.de-en.ffn8192.pt
#DATA_PATH=/home/data_ti5_c/wangdq/data/few_shot/few-shot/data-bin/
#DATASTORE_PATH=/home/wangdq/datastore/few_shot/

DATA_PATH=/home/data_ti6_c/wangdq/data/domain/medical/
DATASTORE_PATH=/home/wangdq/datastore/test/

mkdir -p $DATASTORE_PATH

python /home/data_ti5_c/wangdq/code/few_shot/scripts/knnmt/create_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 1024 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH \
    -s de -t en



python train_datastore_gpu.py \
  --dstore-mmap $DATASTORE_PATH \
  --dstore-size $DSTORE_SIZE \
  --dstore-fp16 \
  --faiss-index ${DATASTORE_PATH}/knn_index \
  --ncentroids 4096 \
  --probe 32 \
  --dimension 1024
