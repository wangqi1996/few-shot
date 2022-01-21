import os.path


def select_sentences(token, src_filename, tgt_filename, write=True, hint=''):
    filter_src, filter_tgt = [], []
    with open(src_filename) as f_src, open(tgt_filename) as f_tgt:
        for src, tgt in zip(f_src, f_tgt):
            tokens = src.strip( ).split(' ')
            if token in tokens:
                filter_src.append(src)
                filter_tgt.append(tgt)
    if write:
        dirname = os.path.dirname(src_filename)
        src_filename = os.path.join(dirname, hint + '.en')
        tgt_filename = os.path.join(dirname, hint + '.zh')
        with open(src_filename, 'w') as f_src:
            f_src.writelines(filter_src)
        with open(tgt_filename, 'w') as f_tgt:
            f_tgt.writelines(filter_tgt)

    return filter_src, filter_tgt


if __name__ == '__main__':
    import sys

    token = sys.argv[1]
    src_filename = sys.argv[2]
    tgt_filename = sys.argv[3]
    write = sys.argv[4]
    if write == 'true':
        write = True
    else:
        write = False
    hint = sys.argv[5]
    select_sentences(token, src_filename, tgt_filename, write, hint)
