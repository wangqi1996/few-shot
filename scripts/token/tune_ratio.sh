cuda=$1

random_seed=32251
ratio=$2

result_path=test

cd /home/wangdq/cwmt/select_10/$random_seed/tune-self
code_dirname=/home/data_ti5_c/wangdq/code/few_shot
python /home/data_ti5_c/wangdq/code/few_shot/scripts/test.py tune-test.txt $ratio
rm -rf ../test
mkdir ../test
cd ../test/
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en ../tune-self/train.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh ../tune-self/ref.zh /home/wangdq/cwmt/codes.zh

fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

echo "pretrain model on tune test-set"
pretrain_model=/home/data_ti6_c/wangdq/few_shot/pretrain-bt/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_model data-bin train

echo "tune model on tune test-set"
tune_model=/home/wangdq/save/wmt17_en2zh/tune-$random_seed/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_model data-bin train

#cd /home/wangdq/cwmt/select_10/$random_seed/pretrain-self
#python /home/data_ti5_c/wangdq/code/few_shot/scripts/test.py pretrain-train.txt $ratio
#rm -rf ../test
#mkdir ../test
#cd ../test/
#/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en ../pretrain-self/train.en /home/wangdq/cwmt/codes.en
#/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh ../pretrain-self/ref.zh /home/wangdq/cwmt/codes.zh
#
#fairseq-preprocess --source-lang en --target-lang zh \
#  --trainpref train.bpe \
#  --srcdict /home/wangdq/cwmt/vocab.en.txt \
#  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
#  --destdir data-bin \
#  --workers 10
#
#echo "pretrain model on pretrain test-set"
#pretrain_model=/home/data_ti6_c/wangdq/few_shot/pretrain-bt/checkpoint_best.pt
#bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_model data-bin train
#
#echo "tune model on pretrain test-set"
#tune_model=/home/wangdq/save/wmt17_en2zh/tune-$random_seed/checkpoint_best.pt
#bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_model data-bin train
