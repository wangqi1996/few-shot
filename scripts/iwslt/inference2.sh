export CUDA_VISIBLE_DEVICES=$1
result_path=$2
model_path=$3
data_path=$4
gen_subset=$5
fairseq-generate $data_path --gen-subset $gen_subset \
  --path $model_path --beam 5 \
  --remove-bpe -s en -t de --max-len-a 1.2 --max-len-b 10 \
  --results-path ~/$result_path/

grep ^D ~/$result_path/generate-$gen_subset.txt | cut -f3- | perl /home/data_ti5_c/wangdq/code/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de >~/$result_path/hypo
grep ^T ~/$result_path/generate-$gen_subset.txt | cut -f2- | perl /home/data_ti5_c/wangdq/code/mosesdecoder/scripts/tokenizer/detokenizer.perl -l de >~/$result_path/ref
cat ~/$result_path/hypo | sacrebleu ~/$result_path/ref
