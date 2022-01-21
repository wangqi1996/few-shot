export CUDA_VISIBLE_DEVICES=$1
result_path=$2
model_path=$3
data_path=$4
gen_subset=train
fairseq-generate $data_path --gen-subset $gen_subset \
  --path $model_path --beam 4 --lenpen 0.6 \
  -s en -t zh --results-path ~/$result_path/
