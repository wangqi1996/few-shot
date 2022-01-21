#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os

from fairseq.data import Dictionary, data_utils


def main(hint, lang, split):
    if hint == 'iwslt':
        dirname = "/home/data_ti5_c/wangdq/data/fairseq/iwslt14/ende/"
    elif hint == "wmt":
        dirname = "/home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq/"

    dict = os.path.join(dirname, "dict." + lang + ".txt")
    input_file = os.path.join(dirname, split + ".en-de." + lang)
    output_file = os.path.join("/home/wangdq/data/" + hint, split + '.' + lang)
    print("dict:", dict)
    print("input:", input_file)
    print("output:", output_file)

    dictionary = Dictionary.load(dict)
    dataset = data_utils.load_indexed_dataset(
        input_file,
        dictionary,
        dataset_impl="mmap",
        default="lazy",
    )
    with open(output_file, 'w') as f:
        for tensor_line in dataset:
            line = dictionary.string(tensor_line, bpe_symbol="@@ ")
            f.write(line + '\n')


if __name__ == "__main__":
    import sys

    dirname = sys.argv[1]
    lang = sys.argv[2]
    split = sys.argv[3]
    main(dirname, lang, split)
