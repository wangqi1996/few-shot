export CUDA_VISIBLE_DEVICES=$1
#DISTILL=/home/wangdq/data/iwslt/data-bin/
DISTILL=/home/wangdq/data/wmt/data-bin/

fairseq-generate $DISTILL \
  --path ~/save/iwslt14_ende/$4/checkpoint_$5.pt \
  --batch-size 128 \
  --beam 5 \
  --remove-bpe \
  --results-path ~/$2 \
  --gen-subset $3 \
  --task translation \
  --seed 1234 \
  -s en -t de \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --model-overrides "{'valid_subset': '$3'}"

tail -1 ~/$2/generate-$3.txt
