export CUDA_VISIBLE_DEVICES=$1

#MODEL_PATH=/home/data_ti6_d/wangdq/wmt17_en2zh/pretrain-all.pt
#DATASET=/home/wangdq/cwmt/temp_select/data-bin/

MODEL_PATH=/home/wangdq/save/wmt17_en2zh/pretrain/checkpoint_best.pt
DATASET=/home/wangdq/cwmt/select_10/oracle-self/data-bin/
result_path=$2
gen_subset=$3

fairseq-generate $DATASET \
  --path $MODEL_PATH --gen-subset $gen_subset \
  --beam 4 --lenpen 0.6 --remove-bpe \
  -s en -t zh \
  --results-path ~/$result_path/ \
#  --sacrebleu-tokenizer zh \
#  --tokenizer moses \
#  --scoring sacrebleu \

grep ^D ~/$result_path/generate-$gen_subset.txt | cut -f3- >~/$result_path/hypo
grep ^T ~/$result_path/generate-$gen_subset.txt | cut -f2- >~/$result_path/ref

cat ~/$result_path/hypo | sacrebleu ~/$result_path/ref --tokenize zh
