export CUDA_VISIBLE_DEVICES=$1

DATASET=/home/data_ti5_c/wangdq/data/few_shot/pretrain/data-bin/
result_path=$2
gen_subset=$3
MODEL_PATH=/home/data_ti6_c/wangdq/few_shot/pretrain-bt/checkpoint_best.pt

fairseq-generate $DATASET \
  --path $MODEL_PATH --gen-subset $gen_subset \
  --beam 4 --lenpen 0.6 --remove-bpe \
  -s en -t zh \
  --scoring sacrebleu \
  --results-path ~/$result_path/ \
  --batch-size 256
#  --tokenizer moses
#  --sacrebleu-tokenizer zh \

grep ^D ~/$result_path/generate-$gen_subset.txt | cut -f3- | perl /home/data_ti5_c/wangdq/code/mosesdecoder/scripts/tokenizer/detokenizer.perl >~/$result_path/hypo
grep ^T ~/$result_path/generate-$gen_subset.txt | cut -f2- | perl /home/data_ti5_c/wangdq/code/mosesdecoder/scripts/tokenizer/detokenizer.perl >~/$result_path/ref

cat ~/$result_path/hypo | sacrebleu ~/$result_path/ref --tokenize zh
