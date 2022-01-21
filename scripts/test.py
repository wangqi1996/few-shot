import os


def select(num=400000):
    import os
    dirname = "/home/wangdq/cwmt/select_10/"
    src_file = os.path.join(dirname, "unused.train.en")
    tgt_file = os.path.join(dirname, "unused.train.zh")
    with open(src_file) as f_src, open(tgt_file) as f_tgt:
        src_content = f_src.readlines()
        tgt_content = f_tgt.readlines()

    all_num = len(src_content)
    import random
    ids = random.sample(range(all_num), num)
    dirname = os.path.join(dirname, "select_40")
    src_file = os.path.join(dirname, "ref.en")
    tgt_file = os.path.join(dirname, "train.zh")
    with open(src_file, 'w') as f_src:
        for id in ids:
            f_src.write(src_content[id])
    with open(tgt_file, 'w') as f_tgt:
        for id in ids:
            f_tgt.write(tgt_content[id])

    src_file = os.path.join(dirname, "unused.train.en")
    tgt_file = os.path.join(dirname, "unused.train.zh")
    # write the unused dataset
    ids = set(ids)
    with open(src_file, 'w') as f_src:
        for index, src in enumerate(src_content):
            if index not in ids:
                f_src.write(src_content[index])
    with open(tgt_file, 'w') as f_tgt:
        for index, tgt in enumerate(tgt_content):
            if index not in ids:
                f_tgt.write(tgt_content[index])


def count_freq(filename, hint, lang):
    token_dict = {}
    with open(filename) as f:
        for line in f:
            for t in set(line.strip().split(' ')):
                t = t.strip()
                if t == "":
                    print(line)
                    print(line.strip().split(' '))
                    continue
                if t in token_dict:
                    token_dict[t] += 1
                else:
                    token_dict[t] = 1
    dirname = os.path.dirname(filename)
    token_dict = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)
    with open(os.path.join(dirname, hint + "dict." + lang + ".txt"), 'w') as f:
        for token in token_dict:
            f.write(token[0] + '\t' + str(token[1]) + '\n')


def read_freq(file, split=' '):
    tokens = {}
    with open(file) as f:
        for line in f:
            if len(line.strip().split(split)) != 2:
                print(line)
            token, freq = line.strip().split(split)
            tokens[token] = int(freq)

    return tokens


def count_bucket_freq_source():
    generate_file = "/home/wangdq/test/generate-test.txt"
    dict_en_txt = "/home/wangdq/cwmt/select_10/dict.en.txt"
    en_freq = read_freq(dict_en_txt, '\t')

    def _count_min_max(min_freq, max_freq):
        hypo_list, ref_list = [], []
        with open(generate_file) as f:
            for line in f:
                if line.startswith("S-"):
                    src = line.strip().split('\t')[1]
                if line.startswith('T-'):
                    tgt = line.strip().split('\t')[1]
                if line.startswith("D-"):
                    hypo = line.strip().split('\t')[2]
                    for token in src.split(' '):
                        freq = en_freq.get(token, 0)
                        if min_freq <= freq < max_freq or (max_freq < 0 and min_freq <= freq):
                            hypo_list.append(hypo)
                            ref_list.append(tgt)
                            break
        import sacrebleu
        bleu_score = sacrebleu.corpus_bleu(hypo_list, [ref_list], tokenize='zh').score
        print("%d ~ %d\t%.2f\t%d" % (min_freq, max_freq, bleu_score, len(hypo_list)))

    _count_min_max(0, -1)
    _count_min_max(1000, -1)
    _count_min_max(500, 1000)
    _count_min_max(100, 500)
    for i in range(100, 0, -10):
        _count_min_max(i - 10, i)


def compare_dict():
    import os
    dirname = "/home/wangdq/cwmt/select_10/"
    test_file, train_file = os.path.join(dirname, "dict.en.txt"), os.path.join(dirname, "unused.dict.en.txt")
    test_freq = read_freq(test_file, '\t')
    train_freq = read_freq(train_file, '\t')
    for token, freq in test_freq.items():
        if freq > 60:
            print(token, freq, train_freq.get(token, 0))
        # if 50 <= freq <= 100 and 300 <= train_freq.get(token, 0) <= 1000:
        #     print(token)


def select_token_by_concurrent():
    # train_filename = "/home/wangdq/cwmt/select_10/train.en"
    # gram = 10
    #
    # def _add(token1, token2):
    #     if token1 == token2:
    #         return
    #
    #     if (token1, token2) in concurrent:
    #         concurrent[(token1, token2)] += 1
    #     elif (token2, token1) in concurrent:
    #         concurrent[(token2, token1)] += 1
    #     else:
    #         concurrent[(token1, token2)] = 1
    #
    # with open(train_filename) as f:
    #     for line in f:
    #         tokens = line.strip().split(' ')
    #         for index in range(len(tokens)):
    #             for i in range(index + 1, min(index + gram + 1, len(tokens))):
    #                 _add(tokens[index], tokens[i])
    #
    # concurrent_filename = os.path.join(os.path.dirname(train_filename), "concurrent.txt")
    # with open(concurrent_filename, 'w') as f:
    #     for (token1, token2), times in concurrent.items():
    #         f.write(token1 + '\t' + token2 + '\t' + str(times) + '\n')

    concurrent_filename = "/home/wangdq/cwmt/select_10/concurrent.txt"
    concurrent = {}
    with open(concurrent_filename) as f:
        for line in f:
            token1, token2, times = line.strip().split('\t')
            concurrent[(token1, token2)] = int(times)

    select = []
    freq = read_freq("/home/wangdq/cwmt/select_10/dict.en.txt", '\t')
    for (token1, token2), times in concurrent.items():
        if times < 3 or times > 15:
            continue
        freq1, freq2 = freq.get(token1, 0), freq.get(token2, 0)
        if 5 < freq1 < 15 and 5 < freq2 < 15:
            print(token1, token2)
            select.append([token1, token2])

    def eq(token1, token2):
        if token1 == token2:
            return True
        return False
        # if token1.lower() == token2.lower() or token1.lower() + 's' == token2.lower() or token1.lower() == token2.lower() + 's':
        #     return True
        # return False

    while True:
        center_token = input()
        candidate = []
        candidate.append(center_token)
        index = 0
        while index < len(candidate):
            center_token = candidate[index]
            for token1, token2 in select:
                if token2 not in candidate and eq(token1, center_token):
                    candidate.append(token2)
                if token1 not in candidate and eq(token2, center_token):
                    candidate.append(token1)
            index += 1
        print(candidate)


"""
rm -rf  tune/
mkdir -p tune/
mv *.select.* tune/
cd  tune/
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.en train.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe train.bpe.zh train.zh /home/wangdq/cwmt/codes.zh
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe dev.bpe.en ../../dev.en /home/wangdq/cwmt/codes.en
/home/data_ti5_c/wangdq/code/fastBPE/fast applybpe dev.bpe.zh ../../dev.zh /home/wangdq/cwmt/codes.zh

fairseq-preprocess --source-lang en --target-lang zh \
    --trainpref train.bpe --validpref ../../dev.bpe --testpref ../../test.bpe \
    --srcdict /home/wangdq/cwmt/vocab.en.txt \
    --tgtdict /home/wangdq/cwmt/vocab.zh.txt \
    --destdir data-bin \
    --workers 10
    
"""

if __name__ == '__main__':
    select_token_by_concurrent()
# compare_dict()
# select_by_token2("conversation")
# select_by_token("conversation", 4, 10)
# select_by_token("Heathrow", 4, 10)
# select_by_token("Olympic", 2, 2)
# select()
# select_by_freq(70, 80, 4)
# count_bucket_freq_source()
# select_by_freq(70, 80)
# count_sentence_freq()
# insert_freq()
# select(400000)
# src_file = "/home/wangdq/cwmt/select_10/train.en"
# src_file = "/home/wangdq/cwmt/test.en"
# count_freq(src_file, 'test.', 'en')
# tgt_file = "/home/wangdq/cwmt/select_10/unused.train.en"
# count_freq(tgt_file, 'unused', 'en')
# src_file = "/home/wangdq/cwmt/select_10/unused.train.en"
# count_freq(src_file, 'unused.')
# select_tokens()
# select_sentences('')
