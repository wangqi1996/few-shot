

export CUDA_VISIBLE_DEVICES=$1
DSTORE_SIZE=100000
shot=$2
DATA_PATH=/home/data_ti5_c/wangdq/data/few_shot/shot/$shot/data-bin/
MODEL_PATH=/home/data_ti6_d/wangdq/pretrain/checkpoint_ave_best.pt
DATASTORE_PATH=/home/wangdq/datastore/$shot/

mkdir -p $DATASTORE_PATH

python /home/data_ti5_c/wangdq/code/few_shot/scripts/knnmt/create_datastore.py $DATA_PATH \
    --dataset-impl mmap \
    --task translation \
    --valid-subset train \
    --path $MODEL_PATH \
    --max-tokens 4096 --skip-invalid-size-inputs-valid-test \
    --decoder-embed-dim 512 --dstore-fp16 --dstore-size $DSTORE_SIZE --dstore-mmap $DATASTORE_PATH \
    -s en -t zh


python /home/data_ti5_c/wangdq/code/few_shot/scripts/knnmt/train_datastore2.py \
  --dstore-mmap $DATASTORE_PATH \
  --dstore-size $DSTORE_SIZE \
  --dstore-fp16 \
  --faiss-index ${DATASTORE_PATH}/knn_index \
  --dimension 512


rm -rf /home/wangdq/datastore/$2/keys.npy