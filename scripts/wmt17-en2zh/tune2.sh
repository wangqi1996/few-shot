export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GUN

#DATA_PATH=/home/wangdq/cwmt/temp_select/data-bin/
#MODEL_PATH=/home/data_ti6_d/wangdq/wmt17_en2zh/pretrain-all.pt

MODEL_PATH=/home/wangdq/save/wmt17_en2zh/pretrain/checkpoint_best.pt
DATA_PATH=/home/wangdq/cwmt/select_10/tune/data-bin

rm -rf ~/save/wmt17_en2zh/tune/

fairseq-train $DATA_PATH \
    --arch transformer_small  --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7e-5 --lr-scheduler fixed \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 16000  --fp16 -s en -t zh \
    --log-format simple --log-interval 100 \
    --max-epoch 10 \
    --num-workers 0  \
    --save-dir ~/save/wmt17_en2zh/tune/  \
    --restore-file $MODEL_PATH   \
    --reset-optimizer --reset-dataloader --reset-meters
