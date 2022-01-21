import os

import numpy as np
from sklearn.cluster import KMeans

n_clusters = 32


def _save(dirname, labels, src_filename, tgt_filename):
    with open(src_filename) as f:
        src_contents = f.readlines()

    with open(tgt_filename) as f:
        tgt_contents = f.readlines()

    os.makedirs(dirname, exist_ok=True)

    for i in range(n_clusters):
        f1, f2 = os.path.join(dirname, str(i) + '.en'), os.path.join(dirname, str(i) + '.de')
        with open(f1, 'w') as f_src, open(f2, 'w') as f_tgt:
            for index, label in enumerate(labels):
                if label == i:
                    f_src.write(src_contents[index])
                    f_tgt.write(tgt_contents[index])


def main():
    dirname = "/home/wangdq/kmeans/iwslt-rep/"
    dstore_size = 160239
    dimension = 512
    rep = np.memmap(os.path.join(dirname, "train.rep"), dtype=np.float16, mode='r', shape=(dstore_size, dimension))

    k_means = KMeans(n_clusters=n_clusters)
    k_means.fit(rep)

    _save(os.path.join(dirname, "cluster-train"), k_means.labels_, os.path.join(dirname, "train.en"),
          os.path.join(dirname, "train.de"))

    def _predict(subset, dstore_size):
        rep = np.memmap(os.path.join(dirname, subset + ".rep"), dtype=np.float16, mode='r',
                        shape=(dstore_size, dimension))

        labels = k_means.predict(rep)
        _save(os.path.join(dirname, "cluster-" + subset), labels, os.path.join(dirname, subset + ".en"),
              os.path.join(dirname, subset + ".de"))

    _predict("valid", 7283)
    _predict("test", 6750)

    import joblib
    joblib.dump(k_means, dirname + 'kmeans.pt')


def test():
    dimension = 512
    dirname = "/home/wangdq/kmeans/wmt-rep/"

    import joblib
    k_means = joblib.load("/home/wangdq/kmeans/iwslt-rep/kmeans.pt")

    def _predict(subset, dstore_size):
        rep = np.memmap(os.path.join(dirname, subset + ".rep"), dtype=np.float16, mode='r',
                        shape=(dstore_size, dimension))

        labels = k_means.predict(rep)
        _save(os.path.join(dirname, "cluster-" + subset), labels, os.path.join(dirname, subset + ".en"),
              os.path.join(dirname, subset + ".de"))

    _predict("train", 3961179)
    _predict("valid", 3000)
    _predict("test", 3003)


if __name__ == '__main__':
    # main()
    test()
