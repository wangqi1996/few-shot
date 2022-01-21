def read_freq(tokens):
    dict_file = "/home/wangdq/cwmt/select_10/dict.en.txt"
    freq1, freq2 = 0, 0
    with open(dict_file) as f:
        for line in f:
            t, freq = line.strip().split('\t')
            if t in tokens:
                freq1 += int(freq)

    dict_file = "/home/wangdq/cwmt/select_10/selec_40/unused.dict.en.txt"
    with open(dict_file) as f:
        for line in f:
            t, freq = line.strip().split('\t')
            if t in tokens:
                freq2 += int(freq)
    print(freq1, freq2)


def read_tokens(token_filename):
    tokens = set()
    with open(token_filename) as f:
        for line in f:
            tokens.add(line.strip())
    return tokens


if __name__ == '__main__':
    import sys

    token_filename = sys.argv[1]
    tokens = read_tokens(token_filename)
    read_freq(tokens)
