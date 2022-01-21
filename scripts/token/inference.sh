
export CUDA_VISIBLE_DEVICES=$1
result_path=$2
model_path=$3
data_path=$4
gen_subset=$5
fairseq-generate $data_path  --gen-subset $gen_subset \
    --path $model_path  --beam 4 --lenpen 0.6 \
    --remove-bpe  -s en -t zh   \
  --results-path ~/$result_path/

#  --tokenizer moses  --sacrebleu-tokenizer zh  --scoring sacrebleu    \

grep ^D ~/$result_path/generate-$gen_subset.txt | cut -f3- > ~/$result_path/hypo
grep ^T ~/$result_path/generate-$gen_subset.txt | cut -f2- > ~/$result_path/ref
cat ~/$result_path/hypo | sacrebleu ~/$result_path/ref  --tokenize zh