import os.path


def select_random_sentences(token, src_filename, tgt_filename, train_num=1, test_num=1):
    filter_src, filter_tgt = [], []
    with open(src_filename) as f_src, open(tgt_filename) as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            tokens = src.strip( ).split(' ')
            if token in tokens:
                filter_src.append(src)
                filter_tgt.append(tgt)

    import random

    train_num = min(train_num, len(filter_src) - test_num)
    ids = random.sample(range(len(filter_tgt)), test_num + train_num)

    index = 0
    dirname = os.path.dirname(src_filename)
    src_filename = os.path.join(dirname, 'train.select.en')
    tgt_filename = os.path.join(dirname, 'train.select.zh')
    with open(src_filename, 'w') as f_src, open(tgt_filename, 'w') as f_tgt:
        while index < train_num:
            f_src.writelines(filter_src[ids[index]])
            f_tgt.writelines(filter_tgt[ids[index]])
            index += 1

    src_filename = os.path.join(dirname, 'test.select.en')
    tgt_filename = os.path.join(dirname,'test.select.zh')
    with open(src_filename, 'w') as f_src, open(tgt_filename, 'w') as f_tgt:
        while index < test_num + train_num:
            f_src.writelines(filter_src[ids[index]])
            f_tgt.writelines(filter_tgt[ids[index]])
            index += 1
    print("train: ", train_num)
    print("test: ", test_num)


if __name__ == '__main__':
    import sys

    token = sys.argv[1]
    src_filename = sys.argv[2]
    tgt_filename = sys.argv[3]
    train_num = int(sys.argv[4])
    test_num = int(sys.argv[5])
    select_random_sentences(token, src_filename, tgt_filename, train_num, test_num)
