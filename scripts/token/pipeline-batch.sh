cuda=$1
random_seed=$RANDOM

code_dirname=/home/data_ti5_c/wangdq/code/few_shot/
data_dirname=/home/wangdq/cwmt/select_10/
tune_data=$data_dirname/${random_seed}/tune/data-bin
oracle_self_data=$data_dirname/${random_seed}/oracle-self/data-bin
select_self_data=$data_dirname/${random_seed}/select-self/data-bin

rm -rf $data_dirname/${random_seed}
mkdir -p $data_dirname/${random_seed}
mv $data_dirname/token.txt $data_dirname/${random_seed}
token_file=$data_dirname/${random_seed}/token.txt
train_num=100000
test_num=1000
pretrain_model=/home/data_ti6_c/wangdq/few_shot/pretrain-bt/checkpoint_best.pt
result_path=test/${random_seed}/

# 统计词频
python $code_dirname/scripts/token/read_freq_batch.py $token_file

# 在10w的pretrain dataset中挑选出含有该token的句子，存入到select_10/token/tune文件夹中  ==> 用来微调的数据集
src_filename=/home/wangdq/cwmt/select_10/train.en
tgt_filename=/home/wangdq/cwmt/select_10/train.zh
python $code_dirname/scripts/token/select_sentences_batch.py $token_file $src_filename $tgt_filename train.select 1 2

mkdir -p $data_dirname/${random_seed}/tune/
cd $data_dirname/${random_seed}/tune/
mv $data_dirname/train.select.* .
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.select.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.select.zh /home/wangdq/cwmt/codes.zh
cp $data_dirname/dev.bpe.* .
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --validpref dev.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

## 在unused数据集中，挑选出含有该token的句子，存入到select_10/token/oracle-self文件夹中  ==> oracle-self数据集
src_filename=/home/wangdq/cwmt/select_10/selec_40/unused.train.en
tgt_filename=/home/wangdq/cwmt/select_10/selec_40/unused.train.zh
python $code_dirname/scripts/token/select_random_sentences_batch.py $token_file $src_filename $tgt_filename $train_num $test_num
mkdir -p $data_dirname/${random_seed}/oracle-self/
cd $data_dirname/${random_seed}/oracle-self/
mv $data_dirname/selec_40/*select* .

/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.select.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.select.zh /home/wangdq/cwmt/codes.zh
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe test.bpe.en test.select.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe test.bpe.zh test.select.zh /home/wangdq/cwmt/codes.zh
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

# select self-training数据集
rm -rf $data_dirname/${random_seed}/select-self/
mkdir -p $data_dirname/${random_seed}/select-self/
cd $data_dirname/${random_seed}/select-self/
bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $tune_data $pretrain_model ${random_seed}

tune_model=/home/wangdq/save/wmt17_en2zh/tune-${random_seed}/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_model $oracle_self_data train
mv ~/$result_path/generate-train.txt tune-train.all.txt
python $code_dirname/scripts/token/select_by_bleu.py ./tune-train.all.txt
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh ref.zh /home/wangdq/cwmt/codes.zh

fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --testpref ../oracle-self/test.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

# 测试pretrain在select数据集上的性能
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_model $select_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $pretrain_model $select_self_data test
mv ~/$result_path/generate-test.txt pretrain-test.txt
mv ~/$result_path/generate-train.txt pretrain-train.txt

bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_model $select_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_model $select_self_data test
mv ~/$result_path/generate-test.txt tune-test.txt
mv ~/$result_path/generate-train.txt tune-train.txt

# 构造self-training数据集
rm -rf $data_dirname/${random_seed}/tune-self/
mkdir -p $data_dirname/${random_seed}/tune-self/
cd $data_dirname/${random_seed}/tune-self/

/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh ../select-self/train.zh /home/wangdq/cwmt/codes.zh
cp ../select-self/train.bpe.en .
cat $data_dirname/${random_seed}/tune/train.bpe.en >>train.bpe.en
cat $data_dirname/${random_seed}/tune/train.bpe.zh >>train.bpe.zh

tune_self_data=$(pwd)/data-bin
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --validpref ../tune/dev.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $tune_self_data $pretrain_model tune-self
tune_self_model=/home/wangdq/save/wmt17_en2zh/tune-tune-self/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $select_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $select_self_data test

# 测试oracl性能
mkdir -p $data_dirname/${random_seed}/oracle
cd $data_dirname/${random_seed}/oracle
cp ../select-self/train.bpe.en .
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh ../select-self/ref.zh /home/wangdq/cwmt/codes.zh
tune_self_data=$(pwd)/data-bin
fairseq-preprocess --source-lang en --target-lang zh \
  --trainpref train.bpe --validpref ../tune/dev.bpe \
  --srcdict /home/wangdq/cwmt/vocab.en.txt \
  --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
  --destdir data-bin \
  --workers 10

bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $tune_self_data $pretrain_model tune-self
tune_self_model=/home/wangdq/save/wmt17_en2zh/tune-tune-self/checkpoint_best.pt
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $select_self_data train
bash $code_dirname/scripts/token/inference.sh $cuda $result_path $tune_self_model $select_self_data test
