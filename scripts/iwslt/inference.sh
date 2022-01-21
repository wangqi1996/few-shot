export CUDA_VISIBLE_DEVICES=$1
DISTILL=/home/data_ti5_c/wangdq/data/fairseq/iwslt14/ende

fairseq-train $DISTILL \
  --ddp-backend=no_c10d \
  --save-dir ~/save/iwslt14_ende/transformer_noshare/ \
  --task translation \
  --arch transformer_iwslt_de_en \
  --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.3 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 --max-tokens 4096 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --no-progress-bar --log-format simple --log-interval 500 \
  --keep-interval-updates 5 \
  --keep-best-checkpoints 5 \
  --no-epoch-checkpoints \
  --save-interval-updates 500 \
  --max-update 100000 \
  -s en -t de \
  --num-workers 4
