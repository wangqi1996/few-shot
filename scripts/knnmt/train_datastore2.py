import argparse
import os
import time

import faiss
import numpy as np

# the implementation refers to knnlm

parser = argparse.ArgumentParser()
parser.add_argument('--dstore-mmap', type=str, help='memmap where keys and vals are stored')
parser.add_argument('--dstore-size', type=int, help='number of items saved in the datastore memmap')
parser.add_argument('--dimension', type=int, default=1024, help='Size of each key')
parser.add_argument('--dstore-fp16', default=False, action='store_true')
parser.add_argument('--faiss-index', type=str, help='file to write the faiss index')

args = parser.parse_args()

print(args)

res = faiss.StandardGpuResources()

print('load dstore fp16', args.dstore_size, args.dimension)
keys = np.memmap(args.dstore_mmap + '/keys.npy', dtype=np.float16, mode='r',
                     shape=(args.dstore_size, args.dimension))
vals = np.memmap(args.dstore_mmap + '/vals.npy', dtype=np.int, mode='r', shape=(args.dstore_size, 1))


if not os.path.exists(args.faiss_index + ".trained"):
    # Initialize faiss index
    index = faiss.IndexFlatL2(args.dimension)
    faiss.write_index(index, args.faiss_index + ".trained")

print('Adding Keys')
index = faiss.read_index(args.faiss_index + ".trained")
co = faiss.GpuClonerOptions()
co.useFloat16 = True
gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
start = 0
num_keys_to_add_at_a_time=500000
start_time = time.time()
while start < args.dstore_size:
    end = min(args.dstore_size, start + num_keys_to_add_at_a_time)
    to_add = keys[start:end].copy()
    gpu_index.add(to_add.astype(np.float32))
    start += num_keys_to_add_at_a_time

    if (start % 1000000) == 0:
        print('Added %d tokens so far' % start)
        print('Writing Index', start)
        faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), args.faiss_index)

print("Adding total %d keys" % end)
print('Adding took {} s'.format(time.time() - start_time))
print('Writing Index')
start = time.time()
faiss.write_index(faiss.index_gpu_to_cpu(gpu_index), args.faiss_index)
print('Writing index took {} s'.format(time.time() - start))