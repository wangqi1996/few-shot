cuda=$1

cluster=4
train_num=10000000
test_num=500
pretrain_model=/home/wangdq/save/iwslt14_ende/pretrain/checkpoint_best.pt
result_path=test
code_dirname=/home/data_ti5_c/wangdq/code/few_shot

srcdict=/home/wangdq/data/iwslt/data-bin/dict.en.txt
tgtdict=/home/wangdq/data/iwslt/data-bin/dict.de.txt
bpe_code=/home/wangdq/data/code

data_dirname=/home/wangdq/kmeans
pretrain_data=$data_dirname/iwslt-rep/cluster
monolingual_data=$data_dirname/wmt-rep/cluster

tune_data=~/tmp/$cluster/tune-data
oracle_data=~/tmp/$cluster/oracle-data
self_data=~/tmp/$cluster/self-data
# 统计词频
echo "iwslt sentences number: "
wc -l $pretrain_data-train/$cluster.de
wc -l $pretrain_data-test/$cluster.de
echo "wmt sentecnes number:"
wc -l $monolingual_data-train/$cluster.en
wc -l $monolingual_data-test/$cluster.en

# tune数据集
mkdir -p $tune_data
cd $tune_data
fairseq-preprocess --source-lang en --target-lang de --trainpref $pretrain_data-train/$cluster \
  --validpref $pretrain_data-valid/$cluster --testpref $pretrain_data-test/$cluster \
  --srcdict $srcdict --tgtdict $tgtdict --destdir data-bin --workers 10

# monolingual数据集
mkdir -p $oracle_data
cd $oracle_data
fairseq-preprocess --source-lang en --target-lang de --trainpref $monolingual_data-train/$cluster \
  --validpref $monolingual_data-valid/$cluster --testpref $monolingual_data-test/$cluster \
  --srcdict $srcdict --tgtdict $tgtdict --destdir data-bin --workers 10

cat $pretrain_data-train/$cluster.en $monolingual_data-train/$cluster.en >train.en
cat $pretrain_data-train/$cluster.de $monolingual_data-train/$cluster.de >train.de
cat $pretrain_data-valid/$cluster.en $monolingual_data-valid/$cluster.en >valid.en
cat $pretrain_data-valid/$cluster.de $monolingual_data-valid/$cluster.de >valid.de
cat $pretrain_data-test/$cluster.en $monolingual_data-test/$cluster.en >test.en
cat $pretrain_data-test/$cluster.de $monolingual_data-test/$cluster.de >test.de

fairseq-preprocess --source-lang en --target-lang de \
  --trainpref train --validpref valid --testpref test \
  --srcdict $srcdict --tgtdict $tgtdict --destdir all --workers 10

#【pretrain】
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $pretrain_model $tune_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $pretrain_model $tune_data/data-bin test
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $pretrain_model $oracle_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $pretrain_model $oracle_data/data-bin test

# 【上限】使用oracle数据训练模型+测试
bash $code_dirname/scripts/iwslt/tune.sh $cuda $oracle_data/all $pretrain_model oracle-$cluster
oracle_model=/home/wangdq/save/iwslt14_ende/oracle-$cluster/checkpoint_last.pt
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $oracle_model $tune_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $oracle_model $tune_data/data-bin test
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $oracle_model $oracle_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $oracle_model $oracle_data/data-bin test

# 【tune】
bash $code_dirname/scripts/iwslt/tune.sh $cuda $tune_data/data-bin $pretrain_model tune-$cluster
tune_model=/home/wangdq/save/wmt17_en2zh/tune-${cluster}/checkpoint_best.pt
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $tune_model $tune_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $tune_model $tune_data/data-bin test
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $tune_model $oracle_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $tune_model $oracle_data/data-bin test

# 【构造self-training数据集】
mkdir -p $self_data
cd $self_data
mv ~/$result_path/generate-train.txt tune-train.txt
grep ^S tune-train.txt | cut -f2- >train.en
grep ^H tune-train.txt | cut -f2- >train.de
python /home/data_ti5_c/wangdq/code/subword-nmt/apply_bpe.py -c $bpe_code <train.en >train.bpe.en
python /home/data_ti5_c/wangdq/code/subword-nmt/apply_bpe.py -c $bpe_code <train.de >train.bpe.de

cat $pretrain_data-train/$cluster/train.en >>train.bpe.en
cat $pretrain_data-train/$cluster/train.de >>train.bpe.de

fairseq-preprocess --source-lang en --target-lang de \
  --trainpref train.bpe --validpref ../oracle_data/valid --testpref ../oracle_data/test \
  --srcdict $srcdict --tgtdict $tgtdict --destdir data-bin --workers 10

bash $code_dirname/scripts/wmt17-en2zh/tune.sh $cuda $self_data/data-bin $pretrain_model self-${cluster}
self_model=/home/wangdq/save/wmt17_en2zh/self-${cluster}/checkpoint_best.pt
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $self_model $tune_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $self_model $tune_data/data-bin test
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $self_model $oracle_data/data-bin train
bash $code_dirname/scripts/iwslt/inference2.sh $cuda $result_path $self_model $oracle_data/data-bin test
