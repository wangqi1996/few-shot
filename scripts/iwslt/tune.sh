export CUDA_VISIBLE_DEVICES=$1
DISTILL=$2
MODEL_PATH=$3
hint=$4

fairseq-train $DISTILL --ddp-backend=no_c10d --save-dir ~/save/iwslt14_ende/${hint}/ \
  --task translation --arch transformer_iwslt_de_en \
  --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
  --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 --max-tokens 4096 \
  --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses --eval-bleu-detok moses --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --no-progress-bar --log-format simple --log-interval 500 \
  --no-epoch-checkpoints \
  -s en -t de --num-workers 4 --max-epoch 10 \
  --restore-file $MODEL_PATH --reset-dataloader --reset-meters
