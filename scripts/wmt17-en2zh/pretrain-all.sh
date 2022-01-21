
export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GUN

fairseq-train /home/data_ti5_c/wangdq/data/few_shot/pretrain/data-bin/ \
    --arch transformer_wmt_en_de     --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 16000     --fp16 -s en -t zh \
    --log-format simple --log-interval 100 \
  --save-interval-updates 500 --keep-best-checkpoints 5 --no-epoch-checkpoints --keep-interval-updates 5 \
  --max-update 300000 \
  --num-workers 0  \
  --save-dir ~/save/wmt17_en2zh/pretrain/  \

