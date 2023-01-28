import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
import torch.nn as nn
import pickle
import argparse
from os import path
from fairseq.modules import (
    LayerNorm,
)

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from pretrain import load_pretrained_model
from modules import init_graphormer_params, GraphormerGraphEncoder

import logging


def parse_args():
    parser = argparse.ArgumentParser()
    # data directory
    parser.add_argument('--encoder_embed_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--num_atoms', type=int, default=512*6)
    parser.add_argument('--num_in_degree', type=int, default=512)
    parser.add_argument('--num_out_degree', type=int, default=512)
    parser.add_argument('--num_edges', type=int, default=512 * 3)
    parser.add_argument('--num_spatial', type=int, default=512)
    parser.add_argument('--num_edge_dis', type=int, default=128)
    parser.add_argument('--edge_type', type=str, default="multi_hop")
    parser.add_argument('--multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--ffn_embedding_dim', type=int, default=256)
    parser.add_argument('--num_attention_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--activation_dropout', type=float, default=0.1)
    parser.add_argument('--encoder_normalize_before', action="store_true")
    parser.add_argument('--pre_layernorm', action="store_true")
    parser.add_argument('--apply_graphormer_init', action="store_true")


class GraphormerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph_encoder = GraphormerGraphEncoder(
                # < for graphormer
                num_atoms=args.num_atoms,
                num_in_degree=args.num_in_degree,
                num_out_degree=args.num_out_degree,
                num_edges=args.num_edges,
                num_spatial=args.num_spatial,
                num_edge_dis=args.num_edge_dis,
                edge_type=args.edge_type,
                multi_hop_max_dist=args.multi_hop_max_dist,
                # >
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.act_dropout,
                encoder_normalize_before=args.encoder_normalize_before,
                pre_layernorm=args.pre_layernorm,
                apply_graphormer_init=args.apply_graphormer_init,
            )
        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = nn.GELU()
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        self.embed_out = nn.Linear(
            args.encoder_embed_dim, args.num_classes, bias=False)

    def forward(self, batched_data):
        inner_states, graph_rep = self.graph_encoder(batched_data)
        x = inner_states[-1].transpose(0, 1)
        x = self.lm_head_transform_weight(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.embed_out(x)
        return x


def eval(args, use_pretrained, checkpoint_path=None, logger=None):
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    # initialize task
    task = tasks.setup_task(cfg.task)
    model = GraphormerModel(args)
    # model = task.build_model(cfg.model)

    # load checkpoint

    model_state = torch.load(checkpoint_path)["model"]
    model.load_state_dict(
        model_state, strict=False
    )
    del model_state

    model.to(torch.cuda.current_device())
    # load dataset
    split = args.split
    task.load_dataset(split)
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=cfg.dataset.max_tokens_valid,
        max_sentences=cfg.dataset.batch_size_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            1000000.0,
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_workers=cfg.dataset.num_workers,
        epoch=0,
        data_buffer_size=cfg.dataset.data_buffer_size,
        disable_iterator_cache=False,
    )
    itr = batch_iterator.next_epoch_itr(
        shuffle=False, set_dataset_epoch=False
    )
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
    )

    # infer
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        with open('D:/Qianqiu/Graphormer/graphormer/criterions/all_entries.pkl',
                  'rb') as f:
            all_entries = pickle.load(f)
        for i, sample in enumerate(progress):

            sample_size = sample["nsamples"]
            left_comps = sample['net_input']['batched_data']['left_comps']
            # print('left_comps', left_comps)
            com_idx = sample['net_input']['batched_data']['com_idx']
            accu_com_idx = torch.cumsum(com_idx, 0)
            # print(sample["net_input"])
            # if self.device.type == "cuda":
            sample["net_input"]['batched_data']["x"] = sample["net_input"]['batched_data']["x"].to(torch.cuda.current_device())
            sample["net_input"]['batched_data']["attn_bias"] = sample["net_input"]['batched_data']["attn_bias"].to(torch.cuda.current_device())
            sample["net_input"]['batched_data']["attn_edge_type"] = sample["net_input"]['batched_data']["attn_edge_type"].to(torch.cuda.current_device())
            sample["net_input"]['batched_data']["spatial_pos"] = sample["net_input"]['batched_data']["spatial_pos"].to(torch.cuda.current_device())
            sample["net_input"]['batched_data']["in_degree"] = sample["net_input"]['batched_data']["in_degree"].to(torch.cuda.current_device())
            sample["net_input"]['batched_data']["out_degree"] = sample["net_input"]['batched_data']["out_degree"].to(torch.cuda.current_device())
            sample["net_input"]['batched_data']["edge_input"] = sample["net_input"]['batched_data']["edge_input"].to(torch.cuda.current_device())
            # print(sample["net_input"]['batched_data'])
            logits = model(**sample["net_input"])[:, 0, :]
            targets = sample["target"]
            # print('com_idx', com_idx)
            for s in range(sample_size):
                invalid_entry = []
                if s == 0:
                    start = 0
                else:
                    start = accu_com_idx[s - 1]
                end = accu_com_idx[s]
                cur_left_composition = left_comps[start:end]
                print('in degree', sample["net_input"]['batched_data']["in_degree"])
                print('out degree', sample["net_input"]['batched_data']["out_degree"])
                print('node feature', sample["net_input"]['batched_data']["x"])
                print('cur_left_composition', cur_left_composition)
                for idx, entry in enumerate(all_entries):
                    for mass in entry:
                        if mass not in cur_left_composition:
                            invalid_entry.append(idx)
                            logits[s, idx] = float('-inf')
                            break
                print('targets[s]', targets[s])
                print('logits[s, :]', logits[s, :])

            sample = utils.move_to_cuda(sample)
            # print('sample', sample)
            y = model(**sample["net_input"])[:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
            torch.cuda.empty_cache()

    # save predictions
    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)

    # evaluate pretrained models


def main():
    parser = options.get_training_parser()
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    args = options.parse_args_and_arch(parser, modify_parser=None)
    logger = logging.getLogger(__name__)
    if args.pretrained_model_name != "none":
        eval(args, True, logger=logger)
    elif hasattr(args, "save_dir"):
        for checkpoint_fname in os.listdir(args.save_dir):
            checkpoint_path = Path(args.save_dir) / checkpoint_fname
            logger.info(f"evaluating checkpoint file {checkpoint_path}")
            eval(args, False, checkpoint_path, logger)


if __name__ == '__main__':
    main()
