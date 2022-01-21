
export CUDA_VISIBLE_DEVICES=$1
#export TOKENIZERS_PARALLELISM=false
#export MKL_THREADING_LAYER=GUN


DATASET=/home/data_ti5_c/wangdq/data/few_shot/pretrain-10/

#DATASET=/home/data_ti5_c/wangdq/data/few_shot/few-shot/data-bin/
#DATASET=/home/data_ti5_c/wangdq/data/few_shot/shot/$3/data-bin/

fairseq-generate $DATASET \
    --path /home/wangdq/save/wmt17_en2zh/pretrain/checkpoint_best.pt \
    --beam 4 --lenpen 0.6 --remove-bpe \
    -s en -t zh \
   --scoring sacrebleu  \
   --sacrebleu-tokenizer zh \
    --results-path ~/$2/ \
    --tokenizer moses

#tail -1 ~/$2/generate-test.txt
grep ^D ~/$2/generate-test.txt | cut -f3- > ~/$2/hypo
grep ^T ~/$2/generate-test.txt | cut -f2- > ~/$2/ref

cat ~/$2/hypo | sacrebleu ~/$2/ref  --tokenize zh
#
python scripts/average_checkpoints.py --inputs ~/save/wmt17_en2zh/pretrain/checkpoint.best_ --output ~/save/wmt17_en2zh/pretrain/checkpoint_ave_best.pt

fairseq-generate $DATASET  \
    --path ~/save/wmt17_en2zh/pretrain/checkpoint_ave_best.pt \
    --beam 4 --lenpen 0.6 --remove-bpe \
    -s en -t zh \
   --scoring sacrebleu  \
   --sacrebleu-tokenizer zh \
    --results-path ~/$2/  \
    --tokenizer moses

#tail -1 ~/$2/generate-test.txt
grep ^D ~/$2/generate-test.txt | cut -f3- > ~/$2/hypo
grep ^T ~/$2/generate-test.txt | cut -f2- > ~/$2/ref

cat ~/$2/hypo | sacrebleu ~/$2/ref  --tokenize zh