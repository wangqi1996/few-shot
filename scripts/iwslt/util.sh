export CUDA_VISIBLE_DEVICES=$1
DISTILL=/home/wangdq/data/iwslt/data-bin/
DISTILL=/home/wangdq/data/wmt/data-bin/

python /home/data_ti5_c/wangdq/code/few_shot/scripts/iwslt/utils.py $DISTILL \
  --valid-subset train \
  --dataset-impl mmap --task translation \
  --path /home/wangdq/save/iwslt14_ende/pretrain/checkpoint_best.pt --max-tokens 16000 \
  --skip-invalid-size-inputs-valid-test --decoder-embed-dim 512 --dstore-size 3961179 \
  --dstore-mmap /home/wangdq/kmeans/wmt-rep/ -s en -t de --num-workers 0 --dstore-fp16

# 160239 train
# 7283 valid
# 6750 test

# 3961179 train
# 3000 valid
# 3003 test
