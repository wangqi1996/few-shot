import os.path
import random


def select_sentences(token, src_filename, tgt_filename, hint='', copy_num=1, random_select=0):
    filter_src, filter_tgt = [], []
    non_select_src, non_select_tgt = [], []
    with open(src_filename) as f_src, open(tgt_filename) as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            tokens = src.strip().split(' ')
            if token in tokens:
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

    token = sys.argv[1]
    src_filename = sys.argv[2]
    tgt_filename = sys.argv[3]
    hint = sys.argv[4]
    copy_num = int(sys.argv[5])
    random_select = int(sys.argv[6])
    select_sentences(token, src_filename, tgt_filename, hint, copy_num, random_select)
