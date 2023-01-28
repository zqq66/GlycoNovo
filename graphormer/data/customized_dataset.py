import collections
import copy
import logging
import torch
import dgl
import multiprocessing
from dgl.data import DGLDataset
import sys
from os import path
from sklearn.model_selection import train_test_split
import pickle
import glypy
import os
import csv
import numpy as np
import scipy.sparse as sp
from glypy.io import glycoct as glypy_glycoct
from glypy.structure.glycan_composition import MonosaccharideResidue, GlycanComposition
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
logger = logging.getLogger(__name__)

from data import register_dataset
from data.eval_util import tree_to_graph, read_spectrum, find_submass
mass_free_reducing_end = 18.01056468370001
mass_proton = 1.00727647

def cut2trees(target_glycan, sugar_classes):
    tree_glycopsm_list = []
    labels = []
    left_compositions = []
    parents = []
    cur_node = target_glycan.index[-1]
    ori_comp = [MonosaccharideResidue.from_monosaccharide(node).mass() for node in target_glycan.iternodes(method='bfs')]
    ori_comp_lst = [sugar_classes.index(mass) for mass in ori_comp]
    ori_comp = collections.Counter(ori_comp_lst)
    ori_comp[len(sugar_classes)-1] = 10
    parents_depth = []
    while cur_node.parents():
        parent = cur_node.parents()[0][-1]
        children = parent.children()
        parent_mass = MonosaccharideResidue.from_monosaccharide(parent).mass()
        parents.append(sugar_classes.index(parent_mass))
        cur_label = torch.zeros((1, len(sugar_classes)), dtype=torch.int32)
        depth = find_depth(parent, target_glycan)
        parents_depth.append(depth)
        for child in children:
            child_pos = child[0]
            parent.drop_monosaccharide(child_pos)
            mass = MonosaccharideResidue.from_monosaccharide(child[1]).mass()
            cur_label[:, sugar_classes.index(mass)] += 1
        labels.append(cur_label)
        target_glycan.reindex(method='bfs')
        tree_glycopsm_list.append(target_glycan.serialize())
        cur_comp = [MonosaccharideResidue.from_monosaccharide(node).mass() for node in target_glycan.iternodes(method='bfs')]
        cur_comp = [sugar_classes.index(mass) for mass in cur_comp]
        cur_comp = collections.Counter(cur_comp)
        ori_comp_copy = copy.deepcopy(ori_comp)
        comp_tensor = torch.zeros((1, len(sugar_classes)))
        for k in ori_comp_copy.keys():
            diff = ori_comp_copy[k]-cur_comp[k]
            comp_tensor[:, k] = diff
        left_compositions.append(comp_tensor)
        cur_node = target_glycan[-1]
    tree_glycopsm_list.append('<sos>')
    last_label = torch.zeros((1, len(sugar_classes)), dtype=torch.int32)
    last_sugar = sugar_classes.index(MonosaccharideResidue.from_monosaccharide(cur_node).mass())
    last_label[:, last_sugar] = 1
    labels.append(last_label)
    parents_depth.append(0)
    ori_comp_tensor = torch.zeros((1, len(sugar_classes)))
    for k in ori_comp.keys():
        ori_comp_tensor[:, k] = ori_comp[k]
    left_compositions.append(ori_comp_tensor)
    parents.append('<sos>')
    return tree_glycopsm_list, labels, left_compositions, parents, parents_depth


class GlycanDBCSV(DGLDataset):
    def __init__(self, glycan_dict, csvfile):
        self.glycan_dict = glycan_dict
        self.graphs = []
        self.labels = []
        self.left_compositions = []
        self.observed_mzs = []
        self.intensities = []
        self.theoretical_mzs = []
        self.csvfile =csvfile
        with open('../criterions/all_entries.pkl', 'rb') as f:
            self.all_entries = pickle.load(f)
        sugar_classes = ['Fuc', 'Hex', 'HexNAc', 'NeuAc', 'NeuGc', 'Xyl']
        stop_token = glypy.monosaccharides['Xyl']
        self.sugar_classes = [glypy.monosaccharides[name].mass() - mass_free_reducing_end for name in sugar_classes]

        self.ion_mass = find_submass(self.all_entries, self.sugar_classes)
        super().__init__(name='glycan_csv')

    def single_fration(self, fraction_id, psm_list, dir):
        for index, psm in enumerate(psm_list[:]):
            psm_scan = psm['Scan']
            scan = 'F' + str(fraction_id) + ':' + psm_scan
            precursor_mass = float(psm['Mass'])
            target_glycan_mass = float(psm['Glycan Mass'])
            adduct_mass = float(psm['Adduct Mass'])
            isotope_shift = float(psm['Isotopic Shift'])
            peptide_only_mass = precursor_mass - target_glycan_mass - adduct_mass + isotope_shift * mass_proton + mass_proton
            target_glycan_id = psm['Glycan ID']
            composition = psm['Glycan']
            glycan = self.glycan_dict[target_glycan_id]['GLYCAN'].clone()
            glycan = glypy_glycoct.loads(glycan.serialize()).reindex(method='bfs')
            tree_glycopsm_list, labels, left_comps, parents, parents_depth = cut2trees(glycan, self.sugar_classes)
            mz, intensity = read_spectrum(scan, peptide_only_mass)
            for idx, tree in enumerate(tree_glycopsm_list[:-1]):
                tree = glypy_glycoct.loads(tree)
                left_comp = torch.tensor(left_comps[idx])
                current_mass = tree.mass() - mass_free_reducing_end + peptide_only_mass
                theoretical_mz = torch.add(current_mass, self.ion_mass)
                graph, adjacency_matrix = tree_to_graph(tree, self.sugar_classes, len(self.sugar_classes), left_comp)
                unordered_label = labels[idx].tolist()

                self.graphs.append(graph)
                self.left_compositions.append(left_comp)
                self.labels.append(self.all_entries.index(unordered_label))
                self.theoretical_mzs.append(torch.tensor(theoretical_mz))
                self.observed_mzs.append(torch.tensor(mz))
                self.intensities.append(torch.tensor(intensity))
            edges_src, edges_dst = np.array([]).astype('int'), np.array([]).astype('int')
            edges_src, edges_dst = np.array([0]).astype('int'), np.array([1]).astype('int')
            root_G = dgl.graph((edges_src, edges_dst), num_nodes=2)
            glycan_lst = composition.replace('(', ',').replace(')', ',').split(',')
            ori_comp_tensor = torch.zeros((1, len(self.sugar_classes)))
            for k, g in enumerate(self.sugar_classes):
                if g in glycan_lst:
                    i = glycan_lst.index(g)
                    ori_comp_tensor[:, k] = int(glycan_lst[i + 1])
            ori_comp_tensor[:, len(self.sugar_classes) - 1] = 10
            special_token = torch.cat((ori_comp_tensor, ori_comp_tensor))

            root_G.ndata['x'] = torch.tensor(special_token,
                                             dtype=torch.int32)  # torch.tensor([len(sugar_classes)], dtype=torch.long)
            self.graphs.append(root_G)
            self.labels.append(self.all_entries.index(labels[-1].tolist()))
            current_mass = peptide_only_mass
            theoretical_mz = torch.add(current_mass, self.ion_mass)
            self.theoretical_mzs.append(torch.tensor(theoretical_mz))
            self.observed_mzs.append(torch.tensor(mz))
            self.intensities.append(torch.tensor(intensity))
            self.left_compositions.append(torch.tensor(left_comps[-1]))
        logger.info("self.graphs| {0}".format(len(self.graphs)))
        with open(dir + "/"+ str(fraction_id) + "left_composition.pkl", 'wb') as fr:
            pickle.dump(self.left_compositions, fr)
        with open(dir + "/"+ str(fraction_id) + "graphs.pkl", 'wb') as fr:
            pickle.dump(self.graphs, fr)
        with open(dir + "/"+ str(fraction_id) + "labels.pkl", 'wb') as fr:
            pickle.dump(self.labels, fr)
        with open(dir + "/"+ str(fraction_id) + "theoretical_mzs.pkl", 'wb') as fr:
            pickle.dump(self.theoretical_mzs, fr)
        with open(dir + "/"+ str(fraction_id) + "observed_mzs.pkl", 'wb') as fr:
            pickle.dump(self.observed_mzs, fr)
        with open(dir + "/"+ str(fraction_id) + "intensities.pkl", 'wb') as fr:
            pickle.dump(self.intensities, fr)

    def process(self):
        self.graphs = []
        self.labels = []
        dir = self.csvfile.split('/')[-1].split('.')[0]
        if not os.path.exists(dir+'/left_composition.pkl'):
            num_fractions = 5
            tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
            fraction_id_list = list(range(1, 1 + num_fractions * (len(tissue_name)+1)))
            glycan_psm = {x: [] for x in fraction_id_list}
            with open(self.csvfile, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    # glycan_score = float(row['Glycan Score'])
                    tissue = row['Source File'].split('-')[0]
                    tissue_id = tissue_name.index(tissue)
                    fraction = int(row['Source File'].split('.')[0][-1])
                    fraction_id = tissue_id * num_fractions + fraction
                    glycan_psm[fraction_id].append(row)
            fraction_id_list = range(1, 21)#[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,21,22, 23, 24, 25]
            print(fraction_id_list)
            procs = []
            for fraction_id in fraction_id_list[:]:
                if not os.path.exists(dir + "/"+ str(fraction_id) + 'left_composition.pkl'):
                    psm_list = glycan_psm[fraction_id]
                    logger.info("fraction_id {0}".format(fraction_id))
                    print('fraction_id', fraction_id)
                    procs.append(
                        multiprocessing.Process(target=self.single_fration, args=(fraction_id, psm_list, dir)))
            for proc in procs:
                proc.start()
            for proc in procs:
                proc.join()

            for fraction_id in fraction_id_list[:]:
                print('fraction_id', fraction_id)
                with open(dir + "/"+ str(fraction_id) + "left_composition.pkl", 'rb') as fr:
                    self.left_compositions += pickle.load(fr)
                with open(dir + "/"+ str(fraction_id) + "graphs.pkl", 'rb') as fr:
                    self.graphs += pickle.load(fr)
                with open(dir + "/"+ str(fraction_id) + "labels.pkl", 'rb') as fr:
                    self.labels += pickle.load(fr)
                with open(dir + "/"+ str(fraction_id) + "theoretical_mzs.pkl", 'rb') as fr:
                    self.theoretical_mzs += pickle.load(fr)
                with open(dir + "/"+ str(fraction_id) + "intensities.pkl", 'rb') as fr:
                    self.intensities += pickle.load(fr)
                with open(dir + "/"+ str(fraction_id) + "observed_mzs.pkl", 'rb') as fr:
                    self.observed_mzs += pickle.load(fr)

        else:
            with open(dir+"/left_composition.pkl", 'rb') as fr:
                self.left_compositions = pickle.load(fr)
            with open(dir+"/graphs.pkl", 'rb') as fr:
                self.graphs = pickle.load(fr)
            with open(dir+"/labels.pkl", 'rb') as fr:
                self.labels = pickle.load(fr)
            with open(dir+"/theoretical_mzs.pkl", 'rb') as fr:
                self.theoretical_mzs = pickle.load(fr)
            with open(dir+"/intensities.pkl", 'rb') as fr:
                self.intensities = pickle.load(fr)
            with open(dir+"/observed_mzs.pkl", 'rb') as fr:
                self.observed_mzs = pickle.load(fr)
        print('ori_comp', self.left_compositions[-1])
        print('len(self.all_entries)', len(self.all_entries))
        print('len(self.graphs)', len(self.graphs))
        print('len(self.labels)', len(self.labels))
        print('len(self.observed_mzs)', len(self.theoretical_mzs))
        print('len(self.mz_intensities)', len(self.intensities))
        print('len(self.left_compositions)', len(self.left_compositions))

    def __getitem__(self, i):
        return self.graphs[i], torch.tensor(self.labels[i]), self.left_compositions[i], self.theoretical_mzs[i], self.observed_mzs[i], self.intensities[i]

    def __len__(self):
        return len(self.graphs)


class GlycanCSV(DGLDataset):
    def __init__(self, glycan_dict, csv_file):
        self.glycan_dict = glycan_dict
        self.graphs = []
        self.labels = []
        self.left_compositions = []
        self.observed_mzs = []
        self.theoretical_mzs = []
        self.intensities = []
        self.csv_file = csv_file
        with open('../criterions/all_entries.pkl', 'rb') as f:
            self.all_entries = pickle.load(f)
        sugar_classes = ['Fuc', 'Hex', 'HexNAc', 'NeuAc', 'NeuGc', 'Xyl']
        stop_token = glypy.monosaccharides['Xyl']
        self.sugar_classes = [glypy.monosaccharides[name].mass() - mass_free_reducing_end for name in sugar_classes]
        self.ion_mass = find_submass(self.all_entries, self.sugar_classes)

        super().__init__(name='glycan_csv')

    def process(self):
        self.graphs = []
        self.labels = []
        num_fractions = 5

        tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
        fraction_id_list = list(range(1, 1 + num_fractions * len(tissue_name)))
        glycan_psm = {x: [] for x in fraction_id_list}
        with open(self.csv_file, 'r') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                # glycan_score = float(row['Glycan Score'])
                tissue = row['Source File'].split('-')[0]
                tissue_id = tissue_name.index(tissue)
                fraction = int(row['Source File'].split('.')[0][-1])
                fraction_id = tissue_id * num_fractions + fraction
                glycan_psm[fraction_id].append(row)

        sugar_classes = ['Fuc', 'Hex', 'HexNAc', 'NeuAc', 'NeuGc', 'Xyl']
        stop_token = glypy.monosaccharides['Xyl']
        num_sugars = len(sugar_classes)
        for fraction_id in fraction_id_list[:]:
            psm_list = glycan_psm[fraction_id]
            for index, psm in enumerate(psm_list[:]):
                psm_scan = psm['Scan']#['ï»¿Scan']
                scan = 'F' + str(fraction_id) + ':' + psm_scan
                try:
                    peptide_only_mass = float(psm['PepMass'])
                except:
                    scan = 'F' + str(fraction_id) + ':' + psm_scan
                    precursor_mass = float(psm['Mass'])
                    target_glycan_mass = float(psm['Glycan Mass'])
                    adduct_mass = float(psm['Adduct Mass'])
                    isotope_shift = float(psm['Isotopic Shift'])
                    peptide_only_mass = precursor_mass - target_glycan_mass - adduct_mass +isotope_shift * mass_proton + mass_proton
                target_glycan_id = psm['Glycan ID']
                if len(target_glycan_id) == 0:
                    target_glycan_id = 0
                composition = psm['Glycan']
                glycan_lst = composition.replace('(', ',').replace(')', ',').split(',')
                ori_comp_tensor = torch.zeros((1, len(sugar_classes)))
                for k, g in enumerate(sugar_classes):
                    if g in glycan_lst:
                        i = glycan_lst.index(g)
                        ori_comp_tensor[:, k] = int(glycan_lst[i+1])
                ori_comp_tensor[:, len(sugar_classes) - 1] = 10
                self.left_compositions.append(ori_comp_tensor)
                edges_src, edges_dst = np.array([0]).astype('int'), np.array([1]).astype('int')
                root_G = dgl.graph((edges_src, edges_dst), num_nodes=2)
                special_token = torch.cat((ori_comp_tensor, ori_comp_tensor))
                root_G.ndata['x'] = torch.tensor(special_token,
                                                 dtype=torch.int32)  # torch.tensor([len(sugar_classes)], dtype=torch.long
                self.graphs.append(root_G)
                self.labels.append(torch.tensor([[int(target_glycan_id), fraction_id, int(psm_scan), peptide_only_mass]]))
                current_mass = peptide_only_mass
                theoretical_mz = torch.add(current_mass, self.ion_mass)
                mz, intensity = read_spectrum(scan, current_mass)
                self.theoretical_mzs.append(torch.tensor(theoretical_mz))
                self.observed_mzs.append(torch.tensor(mz))
                self.intensities.append(torch.tensor(intensity))
        logger.info("self.graphs| {0}".format(len(self.graphs)))

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.left_compositions[i], self.theoretical_mzs[i], self.observed_mzs[i], self.intensities[i]

    def __len__(self):
        return len(self.graphs)


def split_dataset(dataset):
    num_graphs = len(dataset)
    return {
        "dataset": dataset,
        "train_idx": np.arange(num_graphs),
        "valid_idx": np.arange(num_graphs),
        "test_idx": np.arange(num_graphs),
        "source": "dgl"
    }


# @register_dataset("csv_psm")
def create_csv_dataset(csvfile):
    with open('../../../Graphormer/data/glycan_database/glycans_yeast_mouse.pkl', 'rb') as f:
        glycan_dict = pickle.load(f)
    dataset = GlycanCSV(glycan_dict, csvfile)
    return split_dataset(dataset)

def create_psm_db_dataset(csvfile):
    # glycan_dict = read_database()
    with open('../../../Graphormer/data/glycan_database/glycans_yeast_mouse.pkl', 'rb') as f:
        glycan_dict = pickle.load(f)
    dataset = GlycanDBCSV(glycan_dict, csvfile)
    return split_dataset(dataset)


if __name__ == '__main__':
    create_customized_dataset()
