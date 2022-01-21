import logging
import os
import sys

import torch

from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import progress_bar

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    utils.import_user_module(args)

    assert (
            args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    import numpy as np
    print('Saving fp16')
    import os
    os.makedirs(args.dstore_mmap, exist_ok=True)
    dstore_keys = np.memmap(os.path.join(args.dstore_mmap, args.valid_subset + '.rep'), dtype=np.float16, mode='w+',
                            shape=(args.dstore_size, args.decoder_embed_dim))

    dstore_idx = 0
    data_idx = 1
    src_content, tgt_content = [], []
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
    src_pad, tgt_pad = src_dict.pad(), tgt_dict.pad()
    for subset in args.valid_subset.split(","):
        try:
            task.args.required_seq_len_multiple = 1
            task.args.load_alignments = False
            task.load_dataset(subset, combine=False, epoch=data_idx)
            data_idx = data_idx + 1
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        with torch.no_grad():
            model.eval()
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if use_cuda else sample

                features = task.forward_encoder(sample, model)  # [B, T, H]

                # get mean
                src_tokens = sample['net_input']['src_tokens']
                mask = src_tokens == src_pad
                features.masked_fill_(mask.unsqueeze(-1), 0)
                features = features.sum(1)
                length = (~mask).long().sum(-1)
                features = features / length.unsqueeze(1)

                # get plain text
                for i, sample_id in enumerate(sample["id"].tolist()):
                    src, tgt = src_tokens[i], sample['target'][i]
                    src_content.append(src_dict.string(utils.strip_pad(src, src_pad)) + '\n')
                    tgt_content.append(tgt_dict.string(utils.strip_pad(tgt, tgt_pad)) + '\n')

                reduce_size = features.size(0)
                if dstore_idx + reduce_size > args.dstore_size:
                    print("much more than dstore size break")
                    break

                dstore_keys[dstore_idx:reduce_size + dstore_idx] = features.detach().cpu().numpy().astype(
                    np.float16)
                dstore_idx += reduce_size

                print(dstore_idx)

    with open(os.path.join(args.dstore_mmap, args.valid_subset + ".en"), 'w') as f:
        f.writelines(src_content)

    with open(os.path.join(args.dstore_mmap, args.valid_subset + ".de"), 'w') as f:
        f.writelines(tgt_content)


def cli_main():
    parser = options.get_save_datastore_parser()
    args = options.parse_args_and_arch(parser)

    override_parser = options.get_save_datastore_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == "__main__":
    cli_main()
