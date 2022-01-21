import os.path

import sacrebleu


def count_by_bleu(filename):
    src, hypo, tgt = "", "", ""
    result = []
    score_list = []
    with open(filename) as f:
        for line in f:
            if line.startswith('S-'):
                src = line.strip().split('\t')[1]
            if line.startswith('T-'):
                tgt = line.strip().split('\t')[1]
            if line.startswith('D-'):
                hypo = line.strip().split('\t')[2]
                score = sacrebleu.corpus_bleu([hypo], [[tgt]], tokenize="zh").score
                result.append((score, src, hypo, tgt))
                score_list.append(score)
    result.sort(key=lambda t: t[0], reverse=True)
    return result


def select_by_bleu(filename, ratio):
    score = count_by_bleu(filename)
    print("input the min score: ")
    ratio = float(ratio)
    score = score[:int(len(score) * ratio)]
    src_list, hypo_list, tgt_list = [], [], []
    for s, src, hypo, tgt in score:
        src_list.append(src + '\n')
        hypo_list.append(hypo + '\n')
        tgt_list.append(tgt + '\n')

    print(len(src_list))
    dirname = os.path.dirname(filename)
    with open(os.path.join(dirname, "train.en"), 'w') as f:
        f.writelines(src_list)
    with open(os.path.join(dirname, "train.zh"), 'w') as f:
        f.writelines(hypo_list)
    with open(os.path.join(dirname, "ref.zh"), 'w') as f:
        f.writelines(tgt_list)


if __name__ == '__main__':
    import sys

    filename = sys.argv[1]
    ratio = sys.argv[2]
    select_by_bleu(filename, ratio)
