def random_split(src_filename, tgt_filename, train_num, test_num, tgt_dir):
    with open(src_filename) as f:
        src_content = f.readlines()

    with open(tgt_filename) as f:
        tgt_content = f.readlines()

    _len = len(src_content)
    if train_num + test_num > _len:
        train_num = _len - test_num

    import random
    random_ids = random.sample(range(_len), test_num + train_num)
    random.shuffle(random_ids)

    import os
    src_lang, tgt_lang = src_filename[-2:], tgt_filename[-2:]
    src_filename = os.path.join(tgt_dir, "test." + src_lang)
    tgt_filename = os.path.join(tgt_dir, "test." + tgt_lang)
    with open(src_filename, 'w') as f_src, open(tgt_filename, 'w') as f_tgt:
        for id in random_ids[:test_num]:
            f_src.write(src_content[id])
            f_tgt.write(tgt_content[id])

    src_filename = os.path.join(tgt_dir, "train." + src_lang)
    tgt_filename = os.path.join(tgt_dir, "train." + tgt_lang)
    with open(src_filename, 'w') as f_src, open(tgt_filename, 'w') as f_tgt:
        for id in range(_len):
            if id in random_ids[test_num:]:
                f_src.write(src_content[id])
                f_tgt.write(tgt_content[id])


if __name__ == '__main__':
    import sys

    src_filename = sys.argv[1]
    tgt_filename = sys.argv[2]
    train_num = int(sys.argv[3])
    test_num = int(sys.argv[4])
    tgt_dir = sys.argv[5]
    random_split(src_filename, tgt_filename, train_num, test_num, tgt_dir)
