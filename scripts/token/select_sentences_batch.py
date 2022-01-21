import os.path
import random

from scripts.token.read_freq_batch import read_tokens


def select_sentences(select_tokens, src_filename, tgt_filename, hint='', copy_num=1, random_select=0):
    filter_src, filter_tgt = [], []
    non_select_src, non_select_tgt = [], []
    with open(src_filename) as f_src, open(tgt_filename) as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            tokens = src.strip().split(' ')
            select = False
            for token in tokens:
                if token in select_tokens:
                    select = True
                    break
            if select:
                for _ in range(copy_num):
                    filter_src.append(src)
                    filter_tgt.append(tgt)
            else:
                non_select_src.append(src)
                non_select_tgt.append(tgt)

    print(len(filter_src))
    if random_select > 0:
        ids = random.sample(range(len(non_select_src)), random_select * len(filter_src))
        for id in ids:
            filter_src.append(non_select_src[id])
            filter_tgt.append(non_select_tgt[id])

    print(len(filter_src))
    dirname = os.path.dirname(src_filename)
    src_filename = os.path.join(dirname, hint + '.en')
    tgt_filename = os.path.join(dirname, hint + '.zh')
    ids = list(range(len(filter_src)))
    random.shuffle(ids)
    with open(src_filename, 'w') as f_src:
        for id in ids:
            f_src.write(filter_src[id])
    with open(tgt_filename, 'w') as f_tgt:
        for id in ids:
            f_tgt.write(filter_tgt[id])

    return filter_src, filter_tgt


if __name__ == '__main__':
    import sys

    token_filename = sys.argv[1]
    tokens = read_tokens(token_filename)

    src_filename = sys.argv[2]
    tgt_filename = sys.argv[3]
    hint = sys.argv[4]
    copy_num = int(sys.argv[5])
    random_select = int(sys.argv[6])

    # dirname = "/home/wangdq/cwmt/select_10/"
    #
    # tokens = read_tokens(dirname + "26982" + "/token.txt")
    # tokens.update(read_tokens(dirname + "19555" + "/token.txt"))
    # tokens.update(read_tokens(dirname + "20182" + "/token.txt"))
    # tokens.update(read_tokens(dirname + "19622" + "/token.txt"))
    # src_filename = "/home/wangdq/cwmt/select_10/test.en"
    # tgt_filename = "/home/wangdq/cwmt/select_10/test.zh"
    # copy_num = 1
    # random_select = 0
    # hint = ""
    select_sentences(tokens, src_filename, tgt_filename, hint, copy_num, random_select)
