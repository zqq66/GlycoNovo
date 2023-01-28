import torch
import re
from sklearn.model_selection import train_test_split
import csv
import time
import logging
import glypy
import collections
import torch.optim as optim
import numpy as np
import torch.nn as nn
from torch.nn import functional
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
import sys
import dgl
import pickle
import argparse
from os import path
from glypy.io import glycoct as glypy_glycoct

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from modules import GraphormerGraphEncoder
from modules import ionCNN

from data.dgl_datasets.dgl_dataset import GraphormerDGLDataset, preprocess_dgl_graph
from data.customized_dataset import create_csv_dataset, create_psm_db_dataset, create_customized_dataset
from data.collator import collator
from data.eval_util import graph2glycan, test_glycan_accuracy, find_submass, read_csv_files

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
logger = logging.getLogger(__name__)
logger.info("device {0}".format(torch.cuda.current_device()))
mass_free_reducing_end = 18.01056468370001


def parse_args():
    parser = argparse.ArgumentParser()
    # data directory
    parser.add_argument('--encoder_embed_dim', type=int, default=512)
    # parser.add_argument('--num_units', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--num_atoms', type=int, default=512 * 7)
    parser.add_argument('--num_in_degree', type=int, default=512)
    parser.add_argument('--num_sugars', type=int, default=6)
    parser.add_argument('--num_out_degree', type=int, default=512)
    parser.add_argument('--num_edges', type=int, default=512 * 3)
    parser.add_argument('--num_spatial', type=int, default=512)
    parser.add_argument('--num_edge_dis', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--edge_type', type=str, default="multi_hop")
    parser.add_argument('--multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--encoder_layers', type=int, default=2)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--encoder_ffn_embed_dim', type=int, default=256)
    parser.add_argument('--encoder_attention_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--act_dropout', type=float, default=0.1)
    parser.add_argument('--encoder_normalize_before', action="store_true")
    parser.add_argument('--pre_layernorm', action="store_true")
    parser.add_argument(
        "--split",
        type=str,
    )
    parser.add_argument(
        "--metric",
        type=str,
    )
    parser.add_argument("--train_cnn", action='store_true', help="Whether to run training on spectrum.")
    parser.add_argument("--train", action='store_true', help="Whether to run training on glycan structure.")
    parser.add_argument('--inference_cnn', action='store_true', help='Whether to evaluate.')
    parser.add_argument('--prediction', action='store_true', help='Whether to predict.')
    parser.add_argument('--csv_file', type=str, default='../../../Graphormer/data/mouse_tissues.csv')
    parser.add_argument('--mgf_file', type=str, default='../../../Graphormer/data/mouse_tissues_spectrum.mgf')
    parser.add_argument('--graph_model', type=str, default='../../examples/property_prediction/ckpts/model_pos_node_stop.pt')
    parser.add_argument('--cnn_model', type=str, default='../../examples/property_prediction/ckpts/mouse_tissue_all_no_intensity_isotope.pt')
    parser.add_argument('--glycan_db', type=str, default='../../../Graphormer/data/glycan_database/glycans_yeast_mouse.pkl')
    parser.add_argument('--max_time_step', type=int, default=50)

    return parser.parse_args()


class GraphormerIonCNN(nn.Module):
    def __init__(self, args, ion_mass, sugar_classes, graph_embedding=None):
        super().__init__()
        self.graph_embedding = graph_embedding
        self.ionCNN = ionCNN(
            encoder_embed_dim=args.encoder_embed_dim,
            ion_mass=ion_mass,
            sugar_classes=sugar_classes)
        self.embed_out = nn.Linear(
            args.encoder_embed_dim*2, args.num_classes, bias=False)

    def forward(self, batched_data):
        out = self.ionCNN(batched_data)
        with torch.no_grad():
            graph_embed, graph_pred = self.graph_embedding(batched_data)
            combined = torch.cat((graph_embed, out), dim=1)
        out = self.embed_out(combined)
        if not self.training:
            non_zero_idx = (torch.argmax(graph_pred, dim=-1) == 0).nonzero()
            out[non_zero_idx] = graph_pred[non_zero_idx]
        return out


class GraphormerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_sugars = args.num_sugars
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
        )
        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        self.composition_feature_encoder = nn.Embedding(1, args.num_sugars)
        self.composition_type_encoder = nn.Embedding(1, args.num_sugars+1)
        self.depth_weights = nn.Embedding(1, 1)
        self.embed_out = nn.Linear(
            args.encoder_embed_dim, args.num_classes, bias=False)
        # self.softmax = nn.Sigmoid()

    def forward(self, batched_data):
        inner_states, graph_rep = self.graph_encoder(batched_data)
        x = inner_states[-1].transpose(0, 1)
        x = self.lm_head_transform_weight(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        prediction = self.embed_out(x)
        # x = self.softmax(x)
        # print('graph', prediction[:, 0, :])
        return x[:, 0, :], prediction[:, 0, :]


def setup_dataset_torch(args, dataset_dict):

    graphormer_datset = GraphormerDGLDataset(dataset=dataset_dict["dataset"],
                                             train_idx=dataset_dict["train_idx"],
                                             valid_idx=dataset_dict["valid_idx"],
                                             test_idx=dataset_dict["test_idx"],
                                             seed=args.seed)
    # train_idx, val_idx = train_test_split(list(range(len(graphormer_datset))), test_size=0.1)
    # train_set = Subset(graphormer_datset, train_idx)
    val_set = Subset(graphormer_datset, list(range(20)))
    train_dataloader = DataLoader(graphormer_datset,
                                  batch_size=args.batch_size,
                                  collate_fn=lambda x: {key:value.to(device) for key, value in collator(x).items()},
                                  shuffle=True,
                                  )
    val_dataloader = DataLoader(graphormer_datset,
                                batch_size=args.batch_size,
                                collate_fn=lambda x: {key:value.to(device) for key, value in collator(x).items()},
                                shuffle=True,
                                )
    return train_dataloader, val_dataloader


def single_generative_step(parent_node, nodes_onehot, graph):
    graph_feature = graph.ndata['x']
    num_node = graph.number_of_nodes()
    num_new_node = nodes_onehot.shape[0]
    graph.add_nodes(num_new_node)

    graph_feature = torch.cat((graph_feature, nodes_onehot), dim=0).to(torch.int32)
    # print(graph_feature)
    for i in range(num_new_node):
        graph.add_edge([parent_node], [num_node+i])
    graph.ndata['x'] = torch.tensor(graph_feature)

    return graph

def train(model, optimizer, sample, targets, all_entries, graph=True):
    model.train()
    optimizer.zero_grad()
    sample_size = len(sample['idx'])
    left_comps = sample['left_comps']
    sample_dict = {'batched_data': sample}
    if graph:
        logits = model(**sample_dict)[-1]
    else:
        logits = model(**sample_dict)
    targets = targets.to(torch.long)

    torch.cuda.empty_cache()
    loss = functional.cross_entropy(
        logits, targets.reshape(-1), reduction="sum"
    )
    loss.backward()
    optimizer.step()
    left_comps_ = left_comps.repeat(1, 1, len(all_entries)).view(sample_size, len(all_entries), -1)
    all_entries_ = all_entries.repeat(sample_size, 1, 1).view(sample_size, len(all_entries), -1)
    invalid_entry = (left_comps_ - all_entries_) < 0
    invalid_entry_s = torch.any(invalid_entry, -1)
    logits[invalid_entry_s] = float('-inf')
    ncorrect = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()
    return loss, ncorrect, sample_size

def evaluate(model, sample, all_entries):
    model.eval()
    sample_size = len(sample['idx'])
    left_comps = sample['left_comps']
    sample_dict = {'batched_data': sample}
    with torch.no_grad():
        logits = model(**sample_dict)
        targets = sample["y"].to(device)
        torch.cuda.empty_cache()
        loss = functional.cross_entropy(
            logits, targets.reshape(-1), reduction="sum"
        )
    left_comps_ = left_comps.repeat(1, 1, len(all_entries)).view(sample_size, len(all_entries), -1)
    all_entries_ = all_entries.repeat(sample_size, 1, 1).view(sample_size, len(all_entries), -1)
    invalid_entry = (left_comps_ - all_entries_) < 0
    invalid_entry_s = torch.any(invalid_entry, -1)
    logits[invalid_entry_s] = float('-inf')
    ncorrect = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()
    return loss, ncorrect, sample_size


def inference(args, all_entries, model, dataset_dict, sugar_classes, csv_file, denovo=False):
    train_dataloader, val_dataloader = setup_dataset_torch(args, dataset_dict)

    model.eval()

    num_sugars = len(sugar_classes) + 1
    special_token = torch.tensor(len(sugar_classes), device=device)
    complete_graph = []
    target_graph = []
    unable2predict = []
    parent_nodes = dict()
    for i, sample in enumerate(train_dataloader):
        left_comps = sample['left_comps']
        stop_sign = len(sugar_classes) - 1
        with torch.no_grad():
            # print(torch.nonzero(left_comps)[:, -1])
            # full = torch.all(torch.nonzero(left_comps)[:, -1] == stop_sign)
            while not torch.all(torch.nonzero(left_comps)[:, -1] == stop_sign):
                sample_size = len(sample['idx'])
                left_comps = sample['left_comps']
                graphs = dgl.unbatch(sample['graph'])

                ys = sample["y"]
                sample_dict = {'batched_data': sample}
                logits = model(**sample_dict)#[-1]
                # print(logits)
                left_comps_ = left_comps.repeat(1, 1, len(all_entries)).view(sample_size, len(all_entries), -1)
                all_entries_ = all_entries.repeat(sample_size, 1, 1).view(sample_size, len(all_entries), -1)
                invalid_entry = (left_comps_ - all_entries_) < 0
                invalid_entry_s = torch.any(invalid_entry, -1)
                logits[invalid_entry_s] = float('-inf')
                prediction = torch.argmax(logits, dim=-1)
                prediction = all_entries[prediction].view(sample_size, len(sugar_classes))
                # print('prediction', prediction)
                left_comps -= prediction

                new_pyggraphs = []
                for idx, graph in enumerate(graphs):
                    # print(graph.edges(form='uv'))
                    # print(graph.ndata['x'])
                    sample_id = int(sample['idx'][idx])
                    theoretical_mzs = sample['theoretical_mz'][idx]
                    # print('theoretical_mzs', theoretical_mzs)
                    observed_mz = sample['observed_mz'][idx]
                    intensity = sample['intensity'][idx]
                    cur_left_comp = left_comps[idx]
                    if sample_id not in parent_nodes.keys():
                        parent_nodes[sample_id] = [(1, special_token)]

                    parent_node = parent_nodes[sample_id].pop(0)
                    # print(parent_node)

                    new_feature = prediction[idx]
                    num_feature = new_feature.sum()
                    mono_mass = new_feature[:-1].float() @ torch.tensor(sugar_classes[:-1], device=device)
                    theoretical_mzs += mono_mass
                    onehot_newfeature = torch.zeros((num_feature, len(sugar_classes)), dtype=torch.int32, device=device)
                    # print('new_feature', new_feature)

                    for n in range(num_feature - 1):
                        if num_feature == 3:
                            feature = torch.nonzero(new_feature)[0]
                        else:
                            feature = torch.nonzero(new_feature)[-1]
                        new_feature[feature] -= 1
                        onehot_newfeature[n, :] = new_feature
                    # print('onehot_newfeature', onehot_newfeature)
                    new_feature = onehot_newfeature + cur_left_comp
                    # print('new_feature', new_feature)
                    new_graph = single_generative_step(parent_node[0], new_feature, graph)
                    out_degree = new_graph.out_degrees()
                    leaf_nodes = out_degree == 0
                    feature = new_graph.ndata['x'][1:-1] - new_graph.ndata['x'][2:]
                    feature = torch.cat((torch.zeros(2, len(sugar_classes), device=device), feature))
                    valid_entry = feature[:, len(sugar_classes) - 1] != 1
                    valid_entry = torch.logical_and(leaf_nodes, valid_entry)
                    # print('valid_entry', valid_entry)
                    new_parent_node = torch.argmax(new_graph.ndata['x'][valid_entry], dim=1)
                    new_parent_node_idx = torch.nonzero(valid_entry)

                    for i, new_parent in enumerate(new_parent_node_idx):
                        previous_node = [node[0] for node in parent_nodes[sample_id]]
                        if new_parent not in previous_node:
                            parent_nodes[sample_id].append((new_parent, new_parent_node[i]))
                    if torch.all(torch.nonzero(cur_left_comp)[:, -1] == stop_sign):
                        complete_graph.append(graph)
                        target_graph.append(ys[idx])
                    elif len(parent_nodes[sample_id]) == 0:
                        unable2predict.append(int(ys[idx][0]))
                        target_graph.append(ys[idx])
                        complete_graph.append(None)
                        left_comps[idx].fill_(0)
                        continue
                    else:
                        # print('all parents', parent_nodes[sample_id])
                        new_parent_node = parent_nodes[sample_id][0][1]
                        new_pyggraph = preprocess_dgl_graph(new_graph.to('cpu'), ys[idx].view(1, -1), sample_id, cur_left_comp.view(1, len(sugar_classes)), theoretical_mzs, observed_mz, intensity)
                        new_pyggraphs.append(new_pyggraph)
                if new_pyggraphs:
                    sample = {key:value.to(device) for key, value in collator(new_pyggraphs).items()}
    predict_glycans = []
    if denovo:
        glycan_psm = read_csv_files(csv_file)
        predict_csv_name = csv_file.split('.')[0]
        with open(predict_csv_name + '_result.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            rand_psm = list(glycan_psm.keys())[0]
            csvwriter.writerow(list(glycan_psm[rand_psm].keys()) + ['denovo structure'])
            for i, (graph, target) in enumerate(zip(complete_graph, target_graph)):
                if graph:
                    glycan = graph2glycan(graph, sugar_classes_name)
                    predict_glycans.append(glycan)
                    print(glycan)
                    target_glycan, fraction_id, psm_scan, peptide_only_mass = target
                    scan_id = 'F' + str(int(fraction_id)) + ':' + str(int(psm_scan))
                    psm = glycan_psm[scan_id]
                    csvwriter.writerow(list(psm.values()) + [glypy_glycoct.dumps(glycan).replace('\n', ' ')])
    else:
        with open(args.glycan_db, 'rb') as f:
            glycan_dict = pickle.load(f)
        target_glycan = [[glycan_dict[str(int(item[0]))]['GLYCAN'], int(item[1]), int(item[2]), item[3].item()] for item
                         in target_graph]

        print('number of predictions', len(complete_graph))
        for i, graph in enumerate(complete_graph):
            # print(target_graph[i])
            # print(target_glycan[i])
            glycan = graph2glycan(graph, sugar_classes_name) if graph else None
            predict_glycans.append(glycan)
            # print(glycan)
        # print(target_glycan)
        print('unable to predict', len(unable2predict))
        print(unable2predict)
        test_glycan_accuracy(target_glycan, predict_glycans, csv_file)

    return target_glycan, predict_glycans


# with ion CNN encoding spectrum
def train_on_psm(args, all_entries, sugar_classes):
    csvfile = '../../../Graphormer/data/mouse_tissues.csv'
    dataset_dict = create_psm_db_dataset(args, csvfile)
    ion_mass = find_submass(all_entries, sugar_classes)
    train_dataloader, _ = setup_dataset_torch(args, dataset_dict)
    logger.info("\tDone loading dataset")
    model_name = args.graph_model
    print(model_name)
    cnn_model = args.cnn_model
    print(cnn_model)
    graphormer_model = GraphormerModel(args)
    graphormer_model.load_state_dict(torch.load(model_name), strict=False)
    graphormer_model.to(device)

    model = GraphormerIonCNN(args, ion_mass, sugar_classes, graphormer_model)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_ncorrect_val = 0
    end2 = time.time()
    for epoch in range(args.num_epoch):

        train_epoch_loss = 0
        train_ncorrects = 0
        train_sample_sizes = 0

        for i, sample in enumerate(train_dataloader):
            # target = sample["y"]
            target = sample["y"]
            loss, ncorrect, sample_size = train(model, optimizer, sample, target, all_entries, False)
            train_ncorrects += ncorrect
            train_sample_sizes += sample_size
            train_epoch_loss += loss
        train_epoch_loss /= train_sample_sizes
        train_ncorrects = train_ncorrects / train_sample_sizes

        val_epoch_loss = 0
        val_ncorrects = 0
        val_sample_sizes = 0

        if train_ncorrects > best_ncorrect_val:
            logger.info('-Model saved for epoch '.format(epoch) )
            torch.save(model.state_dict(), cnn_model)
            best_ncorrect_val = train_ncorrects
        logger.info("\tTrain Loss: {0}| \tTrain Accuracy: {1}".format(train_epoch_loss, train_ncorrects))
        logger.info("\tVal Loss: {0}| \tVal Accuracy: {1}".format(val_epoch_loss, val_ncorrects))
        print(f'\tTrain Loss: {train_epoch_loss:.3f} | \tVal Loss: {val_epoch_loss:.3f}')
        print(f'\tTrain Accuracy: {train_ncorrects:.3f} | Number of Training Samples: {train_sample_sizes}')
        print(f'\tVal Accuracy: {val_ncorrects:.3f} | Number of Validation Samples: {val_sample_sizes}')


def inference_on_psm(args, all_entries, sugar_classes, csv_file):
    match = re.search(r"_([a-z]+)[0-9]", args.csv_file)
    if match:
        tissue = match.group(1)
    print(tissue)
    print(csv_file)
    dataset_dict = create_csv_dataset(args, csv_file)
    ion_mass = find_submass(all_entries, sugar_classes)
    train_dataloader, val_dataloader = setup_dataset_torch(args, dataset_dict)
    graphormer_model = GraphormerModel(args)
    model_file = '../../examples/property_prediction/ckpts/unseen_'+ tissue+'_graphormer.pt'
    print(model_file)
    graphormer_model.load_state_dict(torch.load(model_file),
                                     strict=False)
    graphormer_model.to(device)

    ion_model = '../../examples/property_prediction/ckpts/mouse_tissue_test_on_unseen_'+tissue+'.pt'
    print(ion_model)
    model = GraphormerIonCNN(args, ion_mass, sugar_classes, graphormer_model)
    model.load_state_dict(torch.load(ion_model), strict=False)
    model.to(device)

    inference(args, all_entries, model, dataset_dict, sugar_classes, csv_file)


def prediction(args, all_entries, sugar_classes, csv_file):
    dataset_dict = create_csv_dataset(args, csv_file)
    ion_mass = find_submass(all_entries, sugar_classes)
    train_dataloader, val_dataloader = setup_dataset_torch(args, dataset_dict)
    graphormer_model = GraphormerModel(args)
    model_file =args.graph_model
    print(model_file)
    graphormer_model.load_state_dict(torch.load(model_file),
                                     strict=False)
    graphormer_model.to(device)
    model = GraphormerIonCNN(args, ion_mass, sugar_classes, graphormer_model)
    model.load_state_dict(torch.load(args.cnn_model),
                          strict=False)
    model.to(device)

    inference(args, all_entries, model, dataset_dict, sugar_classes, csv_file, True)


def train_graph(args, all_entries):
    start_time = time.time()
    dataset_dict = create_customized_dataset()
    train_dataloader, val_dataloader = setup_dataset_torch(args, dataset_dict)

    logger.info("len(all_entris) {0}".format(len(all_entries)))
    end1 = time.time()
    load_dataset_time = end1-start_time
    logger.info("\tDone Loading dataset in {0}".format(load_dataset_time))
    model = GraphormerModel(args)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_ncorrect_val = 0
    end2 = time.time()
    load_model_time = end2 - end1
    for epoch in range(args.num_epoch):

        train_epoch_loss = 0
        train_ncorrects = 0
        train_sample_sizes = 0

        for i, sample in enumerate(train_dataloader):
            loss, ncorrect, sample_size = train(model, optimizer, sample, sample["y"], all_entries)
            train_ncorrects += ncorrect
            train_sample_sizes += sample_size
            train_epoch_loss += loss
        train_epoch_loss /= train_sample_sizes
        train_ncorrects = train_ncorrects / train_sample_sizes
        if train_ncorrects > best_ncorrect_val:
            print('-Model saved for epoch', epoch)
            torch.save(model.state_dict(), args.graph_model)
        logger.info("\tEpoch: {0}".format(epoch))
        logger.info("\tTrain Loss: {0}| \tTrain Accuracy: {1}".format(train_epoch_loss, train_ncorrects))
        # print(f'\tTrain Loss: {train_epoch_loss:.3f} | \tVal Loss: {val_epoch_loss:.3f}' )
        print(f'\tTrain Accuracy: {train_ncorrects:.3f} | Number of Training Samples: {train_sample_sizes}')
        # print(f'\tVal Accuracy: {val_ncorrects:.3f} | Number of Validation Samples: {val_sample_sizes}')
    # evaluate pretrained models
    end3 = time.time()
    training_time = end3-end2

    print('Done Loading model in', load_model_time)
    print('Done Training in', training_time)


if __name__ == '__main__':
    with open('../criterions/all_entries.pkl', 'rb') as f:
        all_entries = pickle.load(f)
    print(all_entries)
    all_entries = torch.tensor(all_entries, device=device)
    sugar_classes_name = ['Fuc', 'Man', 'GlcNAc', 'NeuAc', 'NeuGc', 'Xyl']
    stop_token = glypy.monosaccharides['Xyl']
    args = parse_args()
    sugar_classes = [glypy.monosaccharides[name].mass() - mass_free_reducing_end for name in sugar_classes_name]
    csv_file = args.csv_file
    if args.train:
        train_graph(args, all_entries)
    if args.train_cnn:
        train_on_psm(args, all_entries, sugar_classes)
    if args.inference_cnn:
        inference_on_psm(args, all_entries, sugar_classes, csv_file)
    if args.prediction:
        prediction(args, all_entries, sugar_classes, csv_file)