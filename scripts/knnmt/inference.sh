export CUDA_VISIBLE_DEVICES=$1
DSTORE_SIZE=100000

shot=$2
DATA_PATH=/home/data_ti5_c/wangdq/data/few_shot/shot/$shot/data-bin/
MODEL_PATH=/home/data_ti6_d/wangdq/pretrain/checkpoint_ave_best.pt
DATASTORE_PATH=/home/wangdq/datastore/$shot/

bleu_score=''

for lambda in 0.1 0.2 0.3 0.4 0.5; do
  dirname=~/output/$shot-shot/lambda-${lambda:2:1}

  python fairseq_cli/knn_generate.py $DATA_PATH \
      --gen-subset test \
      --path $MODEL_PATH --arch knn_transformer_wmt19_de_en \
      --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang en --target-lang zh \
      --scoring sacrebleu  --sacrebleu-tokenizer zh \
      --batch-size 128 --fp16 \
      --tokenizer moses --remove-bpe \
      --model-overrides "{'dstore_filename': '$DATASTORE_PATH', 'dstore_size': $DSTORE_SIZE, 'dstore_fp16': True,
      'k': 8, 'knn_lambda_value': $lambda,'knn_temperature_value': 10, 'arch':'knn_transformer_wmt19_de_en'}" \
      --results-path $dirname/

  python scripts/sort.py $dirname/generate-test.txt

  a=$(tail -1 $dirname/generate-test.txt | grep "BLEU = ....." -o)
  a=${a:7:5}
  bleu_score="$bleu_score\n$a"

done
echo $bleu_score