export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GUN

DATA_PATH=$2
MODEL_PATH=$4
hint=$5

fairseq-train $DATA_PATH \
    --arch transformer_iwslt_de_en  --share-decoder-input-output-embed --valid-subset train \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler fixed \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096  --fp16 -s en -t zh \
    --log-format simple --log-interval 100 \
    --max-epoch $3 \
    --num-workers 0  \
    --save-dir ~/save/wmt17_en2zh/tune-${hint}/  \
    --restore-file $MODEL_PATH   \
    --reset-optimizer --reset-dataloader --reset-meters
