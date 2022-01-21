import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from torch_scatter import scatter


class KNN_Dstore(object):

    def __init__(self, args, trg_vocab_size):

        self.half = args.fp16
        self.dimension = args.decoder_embed_dim
        self.dstore_size = args.dstore_size
        self.dstore_fp16 = args.dstore_fp16
        self.vocab_size = trg_vocab_size

        self.index = self.setup_faiss(args)
        self.lambda_value = args.knn_lambda_value
        self.temperature = args.knn_temperature_value
        self.k = args.k

    def setup_faiss(self, args):
        index = faiss.read_index(args.dstore_filename + '/knn_index', faiss.IO_FLAG_ONDISK_SAME_DIR)
        index.nprobe = 32

        res = faiss.StandardGpuResources()
        self.res = res
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

        print('the datastore is {}, size is {}, and dim is {} '.
              format(args.dstore_filename, self.dstore_size, self.dimension))


        self.vals = np.memmap(args.dstore_filename + '/vals.npy', dtype=np.int, mode='r',
                              shape=(self.dstore_size, 1))
        del self.vals
        self.vals_from_memmap = np.memmap(args.dstore_filename + '/vals.npy',
                                          dtype=np.int, mode='r',
                                          shape=(self.dstore_size, 1))
        self.vals = np.zeros((self.dstore_size, 1), dtype=np.int)
        self.vals = self.vals_from_memmap[:]
        self.vals = self.vals.astype(np.int)
        self.vals = torch.from_numpy(self.vals)
        if torch.cuda.is_available():
            print('put vals to gpu')
            self.vals = self.vals.cuda()

        return index


    def get_knns(self, queries):
        dists, knns = self.index.search(queries.float(), self.k)

        return dists, knns

    def retrieve(self, queries):

        bsz = queries.size(0)
        seq_len = queries.size(1)

        dists, knns = self.get_knns(queries.contiguous().view(-1, queries.size(-1)))  # [Batch * seq len, K]

        tgt_idx = self.vals[knns].to(queries.device).squeeze(-1)  # [Batch size * Seq len, K]
        tgt_idx = tgt_idx.view(bsz, seq_len, -1)  # [B, S, K]

        dists = dists.view(bsz, seq_len, -1)  # [Batch, Seq len, k]
        knns = knns.view(bsz, seq_len, -1)

        return {'distance': dists, 'knn_index': knns, 'tgt_index': tgt_idx}


    def calculate_knn_prob(self, tgt_index: torch.Tensor,  # [B, S, K]
                           distance: torch.Tensor):
        bsz, seq_len, _ = distance.size()

        re_compute_dists = -1.0 * distance  # [B, S, K]
        scaled_dists = re_compute_dists / self.temperature
        knn_weight = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)  # [B, S, K, 1]

        knn_tgt_prob = torch.zeros(bsz, seq_len, self.k, self.vocab_size).to(distance.device)  # [B, S, K, Vocab Size]
        tgt_index = tgt_index.unsqueeze(-1)  # [B, S, K, 1]

        scatter(src=knn_weight.float(), out=knn_tgt_prob, index=tgt_index, dim=-1)

        prob = knn_tgt_prob.sum(dim=-2)  # [Batch Size, seq len, vocab size]
        return prob

    def forward(self, hidden_state):
        retrieval = self.retrieve(hidden_state)
        distance, tgt_index = retrieval['distance'], retrieval['tgt_index']
        return self.calculate_knn_prob(tgt_index, distance)