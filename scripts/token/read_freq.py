def read_freq(token):
    dict_file = "/home/wangdq/cwmt/select_10/dict.en.txt"
    with open(dict_file) as f:
        for line in f:
            t, freq = line.strip().split('\t')
            if token == t:
                freq1 = freq

    dict_file = "/home/wangdq/cwmt/select_10/selec_40/unused.dict.en.txt"
    with open(dict_file) as f:
        for line in f:
            t, freq = line.strip().split('\t')
            if token == t:
                freq2 = freq
                print(freq1, freq2)
                return


if __name__ == '__main__':
    import sys

    token = sys.argv[1]
    read_freq(token)
