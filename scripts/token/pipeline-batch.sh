cuda=$1
token=conversation
train_num=10000000
test_num=1000
pretrain_model=/home/wangdq/save/wmt17_en2zh/pretrain/checkpoint_best.pt
result_path=test
code_dirname=/home/data_ti5_c/wangdq/code/few_shot/
data_dirname=/home/wangdq/cwmt/select_10/
tune_data=$data_dirname/${token}/tune/data-bin
oracle_self_data=$data_dirname/${token}/oracle-self/data-bin


# 统计词频
python $code_dirname/scripts/token/read_freq.py $token

rm -rf $data_dirname/${token}/
# 在10w的pretrain dataset中挑选出含有该token的句子，存入到select_10/token/tune文件夹中  ==> 用来微调的数据集
src_filename=/home/wangdq/cwmt/select_10/train.en
tgt_filename=/home/wangdq/cwmt/select_10/train.zh
python $code_dirname/scripts/token/select_sentences.py $token $src_filename $tgt_filename train.select 10 4

mkdir -p $data_dirname/${token}/tune/
cd $data_dirname/${token}/tune/
mv ../../*select* .
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.select.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.select.zh /home/wangdq/cwmt/codes.zh
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe dev.bpe.en ../../../dev.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe dev.bpe.zh ../../../dev.zh /home/wangdq/cwmt/codes.zh
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --validpref dev.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

## 在unused数据集中，挑选出含有该token的句子，存入到select_10/token/oracle-self文件夹中  ==> oracle-self数据集
src_filename=/home/wangdq/cwmt/select_10/unused.train.en
tgt_filename=/home/wangdq/cwmt/select_10/unused.train.zh
python $code_dirname/scripts/token/select_random_sentences.py $token $src_filename $tgt_filename $train_num $test_num
mkdir -p $data_dirname/${token}/oracle-self/
cd $data_dirname/${token}/oracle-self/
mv ../../*select* .

/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.select.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.select.zh /home/wangdq/cwmt/codes.zh
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe test.bpe.en test.select.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe test.bpe.zh test.select.zh /home/wangdq/cwmt/codes.zh
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --testpref test.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

# 测试pretrain在self上的性能，并构造pretrain + self 数据集
rm -rf $data_dirname/${token}/pretrain-self/
mkdir -p $data_dirname/${token}/pretrain-self/
cd $data_dirname/${token}/pretrain-self/

bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_model $oracle_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_model $oracle_self_data test
mv ~/$result_path/generate-test.txt pretrain-test.txt
mv ~/$result_path/generate-train.txt pretrain-train.txt

python $code_dirname/scripts/token/select_by_bleu.py ./pretrain-train.txt
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.zh /home/wangdq/cwmt/codes.zh

cat ../tune/train.bpe.en >>train.bpe.en
cat ../tune/train.bpe.zh >>train.bpe.zh
pretrain_self_data=$(pwd)/data-bin
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --validpref ../tune/dev.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $pretrain_self_data $pretrain_model pretrain-self
pretrain_self_model=/home/wangdq/save/wmt17_en2zh/tune-pretrain-self/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_self_model $oracle_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_self_model $oracle_self_data test

# tune -> 构造pretrain -> tune -> self-training数据集
rm -rf $data_dirname/${token}/tune-self/
mkdir -p $data_dirname/${token}/tune-self/
cd $data_dirname/${token}/tune-self/
bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $tune_data $pretrain_model ${token}

tune_model=/home/wangdq/save/wmt17_en2zh/tune-${token}/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_model $oracle_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_model $oracle_self_data test
mv ~/$result_path/generate-test.txt tune-test.txt
mv ~/$result_path/generate-train.txt tune-train.txt

python $code_dirname/scripts/token/select_by_bleu.py ./tune-train.txt
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.zh /home/wangdq/cwmt/codes.zh

cat ../tune/train.bpe.en >>train.bpe.en
cat ../tune/train.bpe.zh >>train.bpe.zh
pretrain_self_data=$(pwd)/data-bin

tune_self_data=$(pwd)/data-bin
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --validpref ../tune/dev.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $tune_self_data $pretrain_model tune-self
tune_self_model=/home/wangdq/save/wmt17_en2zh/tune-tune-self/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $oracle_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $oracle_self_data test

# 测试oracle性能
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh ref.zh /home/wangdq/cwmt/codes.zh

cat ../tune/train.bpe.zh >>train.bpe.zh
pretrain_self_data=$(pwd)/data-bin

tune_self_data=$(pwd)/data-bin
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --validpref ../tune/dev.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $tune_self_data $pretrain_model tune-self
tune_self_model=/home/wangdq/save/wmt17_en2zh/tune-tune-self/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $oracle_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $oracle_self_data test
