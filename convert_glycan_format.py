import sys

import glypy
import csv
import json
mono_names = ['Man', 'GlcNAc', 'NeuAc', 'NeuGc', 'Fuc']


def convert2glycoCT(structure_encoding):
    idx = 0
    for s in structure_encoding:
        if s.islower():
            structure_encoding = structure_encoding.replace(s, ']')
        elif s.isupper():
            structure_encoding = structure_encoding.replace(s, '[')

    for s in structure_encoding[1:]:
        if s == '[':
            temp_lst = list(structure_encoding)
            temp_lst.insert(idx + 1, ',')
            idx += 2
            structure_encoding = "".join(temp_lst)
        else:
            idx += 1
    struc_lst = json.loads(structure_encoding)

    root = mono_names[struc_lst[0] - 1]
    glycan = glypy.Glycan(root=glypy.monosaccharides[root])
    glycan = construct_glycan(glycan.root, glycan, struc_lst[1:], 0)
    return glycan


def construct_glycan(root, glycan, struc_lst, cur_idx):

    for i, s in enumerate(struc_lst):

        mono = glypy.monosaccharides[mono_names[s[0]-1]]
        root.add_monosaccharide(mono)
        next_idx = cur_idx+ 1
        glycan.reindex(method='dfs')
        root2 = glycan[next_idx]
        construct_glycan(root2, glycan, s[1:], next_idx)
    return glycan


def convert_format(strucgp_file, new_file):
    with open(strucgp_file, 'r') as csvfile:
        with open(new_file, 'w+') as dbfile:
            csvreader = csv.DictReader(csvfile)
            csvwriter = csv.writer(dbfile, delimiter=',')
            for row in csvreader:
                glycan_encoding = row['Structure_codinga']
                glycoct = convert2glycoCT(glycan_encoding)
                row['GlycoCT'] = glypy.io.glycoct.dumps(glycoct).replace('\n', ' ')
                csvwriter.writerow([v for v in row.values()])


if __name__ == '__main__':
    if len(sys.argv) == 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        convert_format(input_file, output_file)
    else:
        print('Wrong number of argument, need to be python3 convert_glycan_format.py input.csv output.csv')

