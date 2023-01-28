import numpy as np
import re
import pickle
import itertools
import torch
import dgl
import glypy
import csv
from glypy.structure.glycan_composition import MonosaccharideResidue, GlycanComposition
from scipy.special import softmax
from glypy.io import glycoct as glypy_glycoct
from torch_geometric.data import Data as PYGGraph
import sys
import collections
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from data.wrapper import convert_to_single_emb
from data import algos


def get_b_y_set(glycan, resolution):

    mass_free_reducing_end = 18.01056468370001
    glycan_clone = glycan.clone()
    glycan_b_set = set()
    glycan_y_set = set()

    for links, frags in itertools.groupby(glycan_clone.fragments(), lambda f: f.link_ids.keys()):
        y_ion, b_ion = list(frags)
        y_mass_reduced = y_ion.mass - mass_free_reducing_end
        b_mass_int = int(round(b_ion.mass * resolution))
        y_mass_int = int(round(y_mass_reduced * resolution))
        glycan_b_set.add(b_mass_int)
        glycan_y_set.add(y_mass_int)
    return glycan_b_set, glycan_y_set


def graph2glycan(graph, sugar_classes):
    num_nodes = graph.number_of_nodes()
    feature = graph.ndata['x'][1:-1] - graph.ndata['x'][2:]
    # print(feature)
    mono_list = torch.argmax(feature, dim=1)
    # print(graph.ndata['x'][1:])
    u, v = graph.edges(form='uv')
    u, v = u[2:] - 2, v[2:] - 2
    # print("u, v", u, v)
    node_dict = {}
    for idx, node in enumerate(u):
        node = int(node)
        if node not in node_dict.keys():
            node_dict[node] = [int(v[idx])]
        else:
            node_dict[node].append(int(v[idx]))
    # print(node_dict)
    # root = glypy.monosaccharides[edge_pairs[0]]
    # root.add_monosaccharide(glypy.monosaccharides[edge_pairs[1]])
    root = sugar_classes[mono_list[0]]
    glycan = glypy.Glycan(root=glypy.monosaccharides[root])
    try:
        for node in range(0, num_nodes-2):
            # print([MonosaccharideResidue.from_monosaccharide(node).residue_name() for node in glycan.iternodes(method='bfs')])
            if node in node_dict.keys():
                for child in node_dict[node]:
                    child_mono = sugar_classes[mono_list[child]]
                    # if child_mono != 'Xyl':
                    parent_mono = glycan[node]
                    parent_mono.add_monosaccharide(glypy.monosaccharides[child_mono])
                    glycan = glycan.reindex(method='bfs')
        leaves = list(glycan.leaves())
        glycan.canonicalize()
        for leaf in leaves:
            if MonosaccharideResidue.from_monosaccharide(leaf).residue_name() == 'Xyl':
                leaf.drop_monosaccharide(-1)
        glycan.canonicalize()
        return glycan
    except:
        return


def read_spectrum(scan_id, peptide_only_mass):
    input_spectrum_file = "../../../Graphormer/data/mouse_tissues_spectrum.mgf"
    spectrum_location_file = input_spectrum_file + '.locations.pkl'
    with open(spectrum_location_file, 'rb') as fr:
        data = pickle.load(fr)
        spectrum_location_dict, spectrum_rtinseconds_dict, spectrum_count = data
    input_spectrum_handle = open(input_spectrum_file, 'r')
    spectrum_location = spectrum_location_dict[scan_id]
    input_file_handle = input_spectrum_handle
    input_file_handle.seek(spectrum_location)

    # parse header lines
    line = input_file_handle.readline()
    assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
    line = input_file_handle.readline()
    assert "TITLE=" in line, "Error: wrong input TITLE="
    line = input_file_handle.readline()
    assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
    line = input_file_handle.readline()
    assert "CHARGE=" in line, "Error: wrong input CHARGE="
    line = input_file_handle.readline()
    assert "RAWFILE" in line, "Error: wrong input RAWFILE="
    line = input_file_handle.readline()
    assert "RAWSCANS" in line, "Error: wrong input RAWSCANS="
    line = input_file_handle.readline()
    assert "SPECGROUPID=" in line, "Error: wrong input SPECGROUPID="
    line = input_file_handle.readline()
    assert "SCANS=" in line, "Error: wrong input SCANS="
    line = input_file_handle.readline()
    assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
    # parse fragment ions
    mz_list = []
    intensity_list = []
    line = input_file_handle.readline()
    while not "END IONS" in line:
        mz, intensity = re.split(' |\n', line)[:2]
        mz_float = float(mz)
        intensity_float = float(intensity)
        # skip an ion if its mass > MZ_MAX
        if mz_float < peptide_only_mass:
            line = input_file_handle.readline()
            continue
        mz_list.append(mz_float)
        intensity_list.append(intensity_float)
        line = input_file_handle.readline()

    return mz_list, intensity_list


def find_submass(all_entries, sugar_classes):
    # device = all_entries.device
    all_entries = torch.tensor(all_entries)
    unique_ions = []
    for idx in range(all_entries.shape[0]):
        new_feature = torch.squeeze(all_entries[idx, :].clone().detach())[:len(sugar_classes)-1]
        num_feature = new_feature.sum()
        onehot_newfeature = torch.zeros(num_feature, dtype=torch.float32)#, device=device)

        for n in range(num_feature):
            feature = torch.nonzero(new_feature)[-1]
            new_feature[feature] -= 1
            onehot_newfeature[n] = sugar_classes[feature]
        all_ions = torch.combinations(onehot_newfeature)
        all_ions = torch.sum(all_ions, dim=-1)
        all_ions = torch.cat((all_ions, onehot_newfeature))
        all_ions = torch.cat((all_ions, torch.sum(onehot_newfeature, dim=-1).unsqueeze(0)))
        unique_ion = torch.unique(all_ions, sorted=True)
        unique_ions.append(unique_ion)
    out = torch.nn.utils.rnn.pad_sequence(unique_ions, batch_first=True)

    return out


def tree_to_graph(tree, sugar_classes, num_sugars, left_comp):
    nodes = dict()
    node_id_to_index = {}
    edges_src = [0, 1]
    edges_dst = [1, 2]
    num_nodes = len(tree.clone().index)
    cur_comp = []
    cur_comp = collections.Counter(cur_comp)
    tree = tree.clone(index_method='bfs')
    for node in tree.index[::-1]:
        node_index = num_nodes-len(nodes)+1
        node_id = node.id
        parents = node.parents()
        if parents:
            node.drop_monosaccharide(parents[0][0])
        node_name = MonosaccharideResidue.from_monosaccharide(node).mass()
        node_sugar_index = sugar_classes.index(node_name)
        comp_tensor = torch.zeros((len(sugar_classes)))

        for k in cur_comp.keys():
            comp_tensor[k] = cur_comp[k]
        cur_comp[node_sugar_index] += 1
        nodes[node_index] = {'id': node_id, 'name': node_name, 'sugar_index': node_sugar_index, 'left_comp': comp_tensor+left_comp}
        node_id_to_index[node_id] = node_index
    num_nodes = len(nodes)
    initial_comp = torch.zeros((len(sugar_classes)))
    for k in cur_comp.keys():
        initial_comp[k] = cur_comp[k]

    adjacency_matrix = np.zeros((num_nodes+2, num_nodes+2))  # np.zeros((num_nodes, num_nodes))
    nodes_onehot = torch.stack([initial_comp+left_comp]*2+[nodes[node]['left_comp'] for node in range(2, num_nodes+2)])
    nodes_onehot = nodes_onehot.view(-1, len(sugar_classes)).to(torch.int32)
    for link in tree.link_index:
        parent_index = node_id_to_index[link.parent.id]
        child_index = node_id_to_index[link.child.id]
        edges_src.append(parent_index)
        edges_dst.append(child_index)
        adjacency_matrix[parent_index, child_index] = 1

    edges_src, edges_dst = np.array(edges_src).astype('int'), np.array(edges_dst).astype('int')
    graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes+2)
    # nodes_onehot = [num_sugars]+ [node['sugar_index'] for node in nodes] # nodes_onehot.astype('float32')
    graph.ndata['x'] = torch.tensor(nodes_onehot, dtype=torch.int32)
    undirected_graph = dgl.add_reverse_edges(graph)
    adjacency_matrix = torch.from_numpy(adjacency_matrix)
    adj_neg = 1 - adjacency_matrix - np.eye(graph.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), graph.number_of_edges())
    train_neg_u, train_neg_v = neg_u[neg_eids[:]].astype('int'), neg_v[neg_eids[:]].astype('int')
    # print(train_neg_u.dtype, train_neg_v.dtype)
    train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())
    train_neg_g.ndata['x'] = torch.tensor(nodes_onehot, dtype=torch.long)

    return graph, adjacency_matrix


def read_csv_files(csvfile):
    # read csv files
    print("Prepare csv_files =", csvfile)
    glycan_psm = {}
    tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
    num_fractions = 5
    with open(csvfile, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            # glycan_score = float(row['Glycan Score'])
            tissue = row['Source File'].split('-')[0]
            tissue_id = tissue_name.index(tissue)
            fraction = int(row['Source File'].split('.')[0][-1])
            fraction_id = tissue_id * num_fractions + fraction
            psm_scan = row['Scan'] # ['ï»¿Scan']
            scan = 'F' + str(fraction_id) + ':' + psm_scan
            glycan_psm[scan] = row

    return glycan_psm


def test_glycan_accuracy(target_glycans, predict_glycans, csvfile, top=None):
    print("test_glycan_accuracy()")

    resolution = 1e3
    num_targets = float(len(target_glycans))
    num_predicts = float(len([x for x in predict_glycans if x]))
    num_target_y = 0.
    num_predict_y = 0.
    num_correct_y = 0.
    correct_glycans = []
    correct_scans = []

    composition_matched = 0
    incorrect_glycan = []
    num_correct_glycans = 0

    composition_incorrect = []
    glycan_psm = read_csv_files(csvfile)
    predict_csv_name = csvfile.split('/')[-1].split('.')[0]
    with open(predict_csv_name + '_denovo.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')

        for idx, (target, candidates) in enumerate(zip(target_glycans, predict_glycans)):
            target_glycan, fraction_id, psm_scan, peptide_only_mass = target
            scan_id = 'F' + str(fraction_id) + ':' + str(psm_scan)
            psm = glycan_psm[scan_id]
            target_b_set, target_y_set = get_b_y_set(target_glycan, resolution)
            num_target_y += len(target_y_set)
            best_predict_y = set()
            best_predict_b = set()
            best_correct_y = set()
            best_correct_glycan = 0.
            best_candidate = None

            candidate = candidates
            predict_b_set, predict_y_set = get_b_y_set(candidate, resolution) if candidate else (set(), set())
            correct_y_set = target_y_set.intersection(predict_y_set)
            correct_glycan = 1 if candidate and candidate.topological_equality(target_glycan) else 0
            best_predict_y = predict_y_set
            best_predict_b = predict_b_set
            best_correct_y = correct_y_set
            best_correct_glycan = correct_glycan
            best_candidate = candidate
            num_predict_y += len(best_predict_y)
            num_correct_y += len(best_correct_y)
            num_correct_glycans += best_correct_glycan
            if best_candidate:
                psm['Glycan ID'] = glypy_glycoct.dumps(best_candidate).replace('\n', ' ')
                csvwriter.writerow([value for value in psm.values()])
            pglyco_lst = [int(MonosaccharideResidue.from_monosaccharide(node).mass())for node in
                          best_candidate.iternodes()] if best_candidate is not None else None
            GF_lst = [int(MonosaccharideResidue.from_monosaccharide(node).mass()) for node in
                      target_glycan.iternodes()]
            pglyco_counter = collections.Counter(pglyco_lst)
            GF_counter = collections.Counter(GF_lst)
            if pglyco_counter == GF_counter:
                composition_matched += 1
            else:
                composition_incorrect.append(scan_id)
                # print(scan_id, pglyco_lst, GF_lst)
            if target_b_set == best_predict_b:
                correct_glycans.append(best_candidate)
                correct_scans.append(scan_id)
            else:
                incorrect_glycan.append((target, best_candidate))
                error_type = 'incorrect_peak'

    sensitivity_y = num_correct_y / num_target_y
    sensitivity_glycan = num_correct_glycans / num_targets
    precision_y = num_correct_y / num_predict_y
    print('incorrect_glycan', incorrect_glycan[-10:])
    print('composition_incorrect', composition_incorrect)
    print('num_correct_compositions', composition_matched)
    print('num_correct_glycans_topologic', num_correct_glycans)
    print("num_targets = ", num_targets)
    print("num_predicts = ", num_predicts)
    print("num_correct_glycans = ", len(correct_glycans))
    print("sensitivity_glycan = {:.2f}".format(sensitivity_glycan))
    print("num_target_y = ", num_target_y)
    print("num_predict_y = ", num_predict_y)
    print("num_correct_y = ", num_correct_y)
    print("sensitivity_y = {:.2f}".format(sensitivity_y))
    print("precision_y = {:.2f}".format(precision_y))
    print('unique_correct_glycan', len(set(correct_glycans)))
    print('correct_scans', correct_scans)
    print('matched on spectrum', spectrum_correct_glycans)
    return num_correct_glycans


