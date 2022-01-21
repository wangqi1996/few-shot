import os.path


def process(file):
    hypos, ref, ids, src = [], [], [], []
    with open(file) as f:
        for line in f:
            if line.startswith("D-"):
                hypos.append(line.split('\t')[-1])
                ids.append(int(line.split('\t')[0][2:]))
            elif line.startswith("T-"):
                ref.append(line.split('\t')[-1])
            elif line.startswith("S-"):
                src.append(line.split('\t')[-1])

    sorted_hypos = [None for _ in ids]
    sorted_ref = [None for _ in ids]
    sorted_src = [None for _ in ids]
    for index, id in enumerate(ids):
        sorted_ref[id] = ref[index]
        sorted_hypos[id] = hypos[index]
        sorted_src[id] = src[index]

    dirname = os.path.dirname(file)
    with open(os.path.join(dirname, 'hypo'), 'w') as f:
        f.writelines(sorted_hypos)

    with open(os.path.join(dirname,'ref'), 'w') as f:
        f.writelines(sorted_ref)


    with open(os.path.join(dirname,'src'), 'w') as f:
        f.writelines(sorted_src)
        
if __name__ == '__main__':
    import sys

    file = sys.argv[1]
    process(file)