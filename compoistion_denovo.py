import argparse
import re
import os
import pickle
import csv
import collections
import glypy
import numpy as np
import time


mass_tolerance = 0.05
isotope_shift = [0]
resolution = 1e2
mass_free_reducing_end = 18.01056468370001
H1_MASS = 1.0078250321
PROTON_MASS = 1.00727567


def parse_args():
    parser = argparse.ArgumentParser()
    # data directory
    parser.add_argument('--mgf_file', type=str, default='C:/shared/mouse_all_tissues/mgf/MouseKidney-Z-T-1.refined.mgf')
    parser.add_argument('--csv_file', type=str, default='C:/shared/mouse_all_tissues/psm/glycanfinder.glycopsms.MouseKidney-Z-T-1.csv')
    parser.add_argument('--output_file', type=str, default='C:/shared/mouse_all_tissues/comp_denovo/all/mouse_kidney_pred_comp1.csv')
    return parser.parse_args()


def merge_mgf_file(input_file_list, fraction_list, output_file):
    """Merge multiple mgf files into one, adding fraction ID to scan ID.

        Usage:
            folder_path = "data.training/aa.hla.bassani.nature_2016.mel_16.class_1/"
            fraction_list = range(0, 10+1)
            merge_mgf_file(
                input_file_list=[folder_path + "export_" + str(i) + ".mgf" for i in fraction_list],
                fraction_list=fraction_list,
                output_file=folder_path + "spectrum.mgf")
    """

    print("merge_mgf_file()")

    # iterate over mgf files and their lines
    counter = 0
    with open(output_file, mode="w") as output_handle:
        for input_file, fraction in zip(input_file_list, fraction_list):
            print("input_file = ", os.path.join(input_file))
            with open(input_file, mode="r") as input_handle:
                for line in input_handle:
                    if "RAWSCANS=" in line:
                        continue
                    elif "SCANS=" in line:  # a spectrum found
                        counter += 1
                        scan = re.split('=|\n|\r', line)[1]
                        # re-number scan id
                        output_handle.write("SCANS=F{0}:{1}\n".format(fraction, scan))

                    else:
                        output_handle.write(line)
    print("output_file = {0:s}".format(output_file))
    print("counter = {0:d}".format(counter))
    print()


# funtion to retrieve spectrum from its scan id
def get_spectrum(input_spectrum_handle, spectrum_location_dict, scan, pep_mass):
    spectrum_location = spectrum_location_dict[scan]
    input_file_handle = input_spectrum_handle
    input_file_handle.seek(spectrum_location)

    # parse header lines
    line = input_file_handle.readline()
    print(line)
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
    signature_idx = None
    line = input_file_handle.readline()
    while not "END IONS" in line:
        mz, intensity = re.split(' |\n', line)[:2]
        mz_float = float(mz)
        intensity_float = float(intensity)
        if signature_idx is None and mz_float > glypy.monosaccharides['NeuGc'].mass() - mass_free_reducing_end:
            signature_idx = len(mz_list)+1
        mz_list.append(mz_float)
        # mz_list.append(int(round(resolution * (mz_float-pep_mass))))
        intensity_list.append(intensity_float)
        line = input_file_handle.readline()
    # input_file_handle.close()
    return mz_list, intensity_list, signature_idx


def spectrum_preprocessing(args, fraction_id_list):

    input_spectrum_file = args.mgf_file
    tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
    num_fractions = 5
    print("Prepare input_spectrum_file =", input_spectrum_file)
    spectrum_location_file = input_spectrum_file + '.locations.pkl'
    if os.path.exists(spectrum_location_file):
        with open(spectrum_location_file, 'rb') as fr:
            print("WorkerIO: read cached spectrum locations")
            data = pickle.load(fr)
            spectrum_location_dict, spectrum_rtinseconds_dict, spectrum_count = data
    else:
        print("WorkerIO: build spectrum location from scratch")
        input_spectrum_handle = open(input_spectrum_file, 'r')
        spectrum_location_dict = {}
        spectrum_rtinseconds_dict = {}
        line = True
        while line:
            current_location = input_spectrum_handle.tell()
            line = input_spectrum_handle.readline()
            if "BEGIN IONS" in line:
                spectrum_location = current_location
            elif "RAWFILE=" in line:
                rawfile = re.split('=|\r|\n|\\\\', line)[-2]
                print(rawfile)
                tissue = rawfile.split('-')[0]
                tissue_id = tissue_name.index(tissue)
                fraction = int(rawfile.split('.')[0][-1])
                fraction_id = tissue_id*num_fractions+fraction
                print(fraction_id)
            elif "SCANS=" in line:
                scan = re.split('=|\r|\n', line)[1]
                scan = 'F' + str(fraction_id) + ':' + scan
                spectrum_location_dict[scan] = spectrum_location
            elif "RTINSECONDS=" in line:
                rtinseconds = float(re.split('=|\r|\n', line)[1])
                spectrum_rtinseconds_dict[scan] = rtinseconds
        spectrum_count = len(spectrum_location_dict)
        with open(spectrum_location_file, 'wb') as fw:
            pickle.dump((spectrum_location_dict, spectrum_rtinseconds_dict, spectrum_count), fw)
        input_spectrum_handle.close()
    print("len(spectrum_location_dict) =", len(spectrum_location_dict))
    print()
    return input_spectrum_file, spectrum_location_dict


# prediction should read unlabeled data
def read_csv_files(args, fraction_id_list):
    # read csv files
    csvfile = args.csv_file
    print("Prepare csv_files =", csvfile)
    glycan_psm = {x: [] for x in fraction_id_list}
    with open(csvfile, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            # glycan_score = float(row['Glycan Score'])
            fraction_id = int(row['Source File'].split('.')[0][-1])
            glycan_psm[fraction_id].append(row)
    total_psm = 0
    for fraction_id, psm_list in glycan_psm.items():
        print("fraction_id =", fraction_id, ",", "len(psm_list) =", len(psm_list))
        total_psm += len(psm_list)
    print("total_psm =", total_psm)
    print()
    return glycan_psm


def preprocess(glycan):
    order = ['Fuc', 'Hex', 'HexNAc', 'NeuAc', 'NeuGc']
    exist_mono = []
    glycan = glycan.split('%')[0].replace('(', ',').replace(')', ',').split(',')
    converted_glycan = []
    for idx, g in enumerate(order):
        if g in glycan:
            exist_mono.append(g)
            i = glycan.index(g)
            converted_glycan += [(idx+1) for _ in range(int(glycan[i + 1]))]
    gc = collections.Counter(converted_glycan)
    string = ''
    for i in range(1, 6):
        string += str(i) + '('
        string += str(gc[i]) + ')'
    return string, exist_mono


def find_matched_peak_within_tol_np(theoretical_mass, score_table):
    mz1_list = list(score_table.keys())
    matched = np.isclose(mz1_list, theoretical_mass, atol=mass_tolerance, rtol=0)
    matched_idx = np.nonzero(matched)
    if len(matched_idx[0]) == 0:
        return 0, 0
    elif len(matched_idx[0]) > 1:
        idx = np.argmin(abs(mz1_list[i]-theoretical_mass) for i in matched_idx[0])
        matched_mass = mz1_list[matched_idx[0][idx]]
        print('matched_mass', matched_mass)
    else:
        matched_mass = mz1_list[matched_idx[0][0]]
    return score_table[matched_mass], matched_mass


def find_matched_peak_within_tol(theoretical_mass, score_table):
    # find the closest peak to the theoretical mass and return its intensity
    # TODO: tolerance level find the closest peak
    # TODO: isotope shift find the highest peak
    num_accepted_shift = len(isotope_shift)
    mass_range = []
    mz1_list = list(score_table.keys())
    num_mz0 = len(mz1_list)
    mz0_array = np.array(mz1_list)
    mz0_array = np.broadcast_to(mz0_array, shape=(num_accepted_shift, num_mz0))
    mz0_array = np.transpose(mz0_array)
    for shift in isotope_shift:
        sub_peak = int(round((theoretical_mass/resolution + shift) * resolution))
        mass_range.append(sub_peak)
    glycopeptide_array = np.array(mass_range)
    glycopeptide_array = np.broadcast_to(glycopeptide_array, shape=(num_mz0, num_accepted_shift))
    delta = np.abs(mz0_array - glycopeptide_array)
    # find the index of the first non-zero entries
    counts = np.count_nonzero(delta <= mass_tolerance * resolution, axis=0)
    sum_counts = np.sum(counts)
    if sum_counts != 0:
        index = np.nonzero(counts)[0].tolist()
        if len(index) > 1:
            max_intensity = float('-inf')
            matched_index = None
            for i in index:
                current_index = np.nonzero(delta[:, i] <= mass_tolerance * resolution)[0][0]
                current_intensity = score_table[mz1_list[current_index]]
                if current_intensity > max_intensity:
                    max_intensity = current_intensity
                    matched_index = current_index
        else:
            matched_index = np.nonzero(delta[:, index[0]] <= mass_tolerance * resolution)[0][0]
        mass = mz1_list[matched_index]
        return score_table[mass], mass
    return 0, 0


def construct_table(peptide_only_mass, mz1_list, intensity_list, sugar_classes):
    num_glyco = len(sugar_classes) * len(isotope_shift)
    peak_table = dict(zip(mz1_list, intensity_list))
    peak_table = collections.OrderedDict(sorted(peak_table.items()))
    num_mz0 = len(mz1_list)
    mz0_array = np.array(mz1_list)
    mz0_array = np.broadcast_to(mz0_array, shape=(num_glyco * num_mz0, num_mz0))
    mz0_array = np.transpose(mz0_array)
    peaks = []
    max_mono = max(sugar_classes)
    for peak in sorted(peak_table.keys()):
        print('peak', peak)
        sub_peaks = []
        for monosaccharide in sugar_classes:
            for shift in isotope_shift:
                sub_peak = int(round((peak/resolution - monosaccharide + shift) * resolution))
                sub_peaks.append(sub_peak)
        peaks += sub_peaks
    glycopeptide_array = np.array(peaks)
    glycopeptide_array = np.broadcast_to(glycopeptide_array, shape=(num_mz0, num_glyco * num_mz0))
    print(mz0_array.shape)
    print(glycopeptide_array.shape)
    delta = np.abs(mz0_array - glycopeptide_array)
    counts = np.count_nonzero(delta <= mass_tolerance * resolution, axis=0)
    matched_peak = sum(counts)
    dp_table = collections.OrderedDict((i, float('-inf')) for i in peak_table.keys())
    dp_table[int(round(peptide_only_mass * resolution))] = 0

    print(matched_peak)


def dp(glycan_mass, pep_mass, mz1_list, intensity_list, sugar_classes):
    glycan_mass = int(round((glycan_mass+mass_tolerance) * resolution))
    score_table = dict()
    for idx, mz in enumerate(mz1_list):
        if mz-pep_mass >= 0:
            score_table[int(round(resolution * (mz-pep_mass)))] = np.log(intensity_list[idx])
    print(score_table.keys())
    if len(score_table.keys()) == 0 or (glycan_mass - max(score_table.keys())) / resolution > 1000:
        return None
    print('len(score_table.keys())', len(score_table.keys()))
    glycan_range = int(round(glycan_mass * resolution))
    dp_table = np.full((glycan_range, ), np.NINF)
    composition_table = []
    sugar_classes = [int(round(monomass * resolution)) for monomass in sugar_classes]
    # TODO: initialize 0 within tolerance level = 0
    for i in isotope_shift:
        res = int(round(i*resolution))
        dp_table[abs(res)] = 0
        dp_table[abs(res+1)] = 0
        dp_table[abs(res+2)] = 0
        dp_table[abs(res+3)] = 0
    current_mass = 0
    tol_time = 0
    while current_mass <= glycan_mass:
        start = time.time()
        current_inensity, _ = find_matched_peak_within_tol(current_mass, score_table)
        end = time.time()
        tol_time += end-start
        prev_max_intensity = float('-inf')
        prev_path = []

        for idx, monomass in enumerate(sugar_classes):
            penalty = 1
            if idx == 3:
                penalty = 1.1
            elif idx == 4:
                penalty = 1.1
            submass = current_mass - monomass
            if submass < -2*resolution:
                continue
            submass = max(0, submass)
            prev_intensity = dp_table[submass] * penalty
            if prev_intensity > prev_max_intensity:
                # print('prev_intensity', prev_intensity)
                prev_max_intensity = prev_intensity
                composition = idx + 1
                prev_path =[(composition, submass)]
            elif prev_intensity == prev_max_intensity and prev_intensity > 0:
                composition = idx + 1
                prev_path.append((composition, submass))

        updated_intensity = prev_max_intensity + current_inensity
        if updated_intensity >= 0:
            dp_table[current_mass] = updated_intensity
        composition_table.append(prev_path)
        # print('dp_table[current_mass]', dp_table[current_mass])
        # print(' composition_table[current_mass]',  composition_table[current_mass])
        current_mass += 1

    print('find_matched_peak_within_tol', tol_time)

    return dp_table, composition_table


def traceback(args, target_glycan_mass, sugar_classes, dp_table, composition_table):
    glycan_mass = int(round((target_glycan_mass) * resolution))
    max_index = glycan_mass
    if len(composition_table[max_index]) == 0:
        return
    paths = get_paths(composition_table, (0, max_index))
    # print('paths', paths)
    actual_path = [paths[-1]]
    for idx, path in enumerate(paths):
        if idx+1 < len(paths) and len(path) > len(paths[idx+1]):
            actual_path.append(path)  # [i[0] for i in path])
    del paths
    compositions = []
    last_nodes = []
    for path in actual_path:
        # print([(i[0], dp_table[i[1]]) for i in path[1:]])
        path = [i[0] for i in path[1:]]
        pc = collections.Counter(path)
        mass = round(sum([sugar_classes[i-1] * pc[i] for i in pc.keys()]), 2)
        string = ''
        for i in range(1, 6):
            string += str(i) + '('
            string += str(pc[i]) + ')'
        # print(string)
        if (string, mass) not in compositions:
            compositions.append((string, mass))
            last_nodes.append(path[-1])
    del actual_path
    return set(compositions), last_nodes


def get_paths(composition_table, tree):
    if tree[1] == 0:
        return [[tree]]
    else:
        root = tree
        rooted_paths = [[root]]
        for subtree in composition_table[tree[1]]:
            paths = get_paths(composition_table, subtree)
            for path in paths:
                rooted_paths.append([root]+path)
        return rooted_paths


def find_signature_ion(mz1_list, intensity_list, signature_idx):
    sum_intensity = max(intensity_list)
    relative_intensity_list = [i/sum_intensity * 100 for i in intensity_list]
    sugar_classes = ['NeuAc', 'NeuGc']
    sugar_mass = [glypy.monosaccharides[mono].mass() - mass_free_reducing_end + PROTON_MASS for mono in sugar_classes]
    sugar_no_H2O = [glypy.monosaccharides[mono].mass() - mass_free_reducing_end *2 + PROTON_MASS for mono in sugar_classes]
    existed_ion = []
    for idx, mono in enumerate(sugar_classes):
        existence = np.isclose(mz1_list[:signature_idx], sugar_mass[idx], atol=3*1e-2)
        pure_int = sum([relative_intensity_list[i] for i in np.nonzero(existence)[0]])
        existence_H2O = np.isclose(mz1_list[:signature_idx], sugar_no_H2O[idx], atol=3*1e-2)
        loss_H2O = sum([relative_intensity_list[i] for i in np.nonzero(existence_H2O)[0]])
        if pure_int > 0.5 and loss_H2O > 0.5:
            existed_ion.append(mono)
    return existed_ion


def unbound_composition(sugar_class, composition):
    composition = composition.split('(')
    lst = []
    for i in composition:
        lst += i.split(')')
    result = ''
    dic = {}
    for i in range(len(sugar_class)):
        mono = sugar_class[i]
        dic[mono] = int(lst[2*i+1])
        if int(lst[2*i+1]) > 0:
            result += '('+ mono+')'+str(lst[2*i+1])
    return result, dic


def main(args):
    num_fractions = 5
    fraction_id_list = list(range(1, 1 + num_fractions*5))
    input_spectrum_file, spectrum_location_dict = spectrum_preprocessing(args, fraction_id_list)
    input_spectrum_handle = open(input_spectrum_file, 'r')
    glycan_psm = read_csv_files(args, fraction_id_list)
    num_correct_comp = 0
    num_predicted = 0
    incorrect_comp_scan = []
    no_composition_scan = []

    multiple_prediction = 0
    with open(args.output_file, 'w', newline='') as csvfile:
        rand_psm = list(glycan_psm.keys())[0]
        csvwriter.writerow(list(glycan_psm[rand_psm].keys())
        csvwriter = csv.writer(csvfile, delimiter=',')

        for fraction_id in fraction_id_list[:]:
            psm_list = glycan_psm[fraction_id]
            for index, psm in enumerate(psm_list[:]):
                tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
                # tissue_name = ['ShenJ_FourStandardGlycoproteins_CE20_33_Run1.raw']
                tissue = psm['Source File'].split('-')[0]
                tissue_id = tissue_name.index(tissue)
                fraction_id = int(psm['Source File'].split('.')[0][-1])
                fraction_id = tissue_id * num_fractions + fraction_id
                scan = 'F' + str(fraction_id) + ':' + psm['Scan'] #psm['ï»¿Scan']
                print('scan', scan)
                precursor_mass = float(psm['Mass'])
                target_glycan_mass = float(psm['Glycan Mass'])
                adduct_mass = float(psm['Adduct Mass'])
                peptide_only_mass = precursor_mass - target_glycan_mass - adduct_mass + PROTON_MASS
                # peptide_only_mass = float(psm['PepMass'])
                glycan = psm['Glycan']
                target_glycan, exist_mono = preprocess(glycan)
                # print('target_glycan', target_glycan)
                mz1_list, intensity_list, signature_idx = get_spectrum(input_spectrum_handle, spectrum_location_dict, scan, peptide_only_mass)
                start = time.time()
                sugar_class = ['Fuc', 'Hex', 'HexNAc'] + find_signature_ion(mz1_list, intensity_list, signature_idx)
                print('sugar_classes', sugar_class)
                sugar_classes = [glypy.monosaccharides[mono].mass() - mass_free_reducing_end for mono in sugar_class]
                print(sugar_classes)
                dp_result = dp(target_glycan_mass, peptide_only_mass, mz1_list, intensity_list, sugar_classes)
                if not dp_result:
                    continue
                dp_table, composition_table = dp_result

                end1 = time.time()
                construct_time = end1-start
                print('Table  construct', construct_time)
                traceback_result = traceback(args, target_glycan_mass, sugar_classes, dp_table, composition_table)
                if not traceback_result:
                    continue
                compositions, last_nodes = traceback_result
                # paths.append(compositions)
                end2 = time.time()
                traceback_time = end2-end1
                print('Traceback time', traceback_time)
                all_time = end2-start
                print('Overall time', all_time)
                true_mass_comp = []
                true_comp = 0
                if len(compositions) == 0:
                    no_composition_scan.append(scan)
                elif len(compositions) > 1:
                    multiple_prediction += 1

                best_compoistion = {'Fuc': 10}
                best_str = None
                for idx, (composition, mass) in enumerate(compositions):
                    if target_glycan_mass-2 < mass < target_glycan_mass+2:
                        result, dic = unbound_composition(sugar_class, composition)
                        print('dic', dic)
                        if best_compoistion['Fuc'] > dic['Fuc'] and dic['Hex'] >= 3 and dic['HexNAc'] >= 2:
                            best_compoistion = dic
                            best_str = result
                        true_mass_comp.append(composition)
                if best_str:
                    target_str, _ = unbound_composition(sugar_class, target_glycan)
                    psm['Glycan'] = best_str
                    print('best_str', best_str)
                    csvwriter.writerow([value for value in psm.values()])
                    if best_str == target_str:
                        num_correct_comp += 1
                        true_comp += 1
                    else:
                        incorrect_comp_scan.append(scan)

                    num_predicted += len(compositions)
                    print('len(true_mass_comp)', len(true_mass_comp))
                    print('true_comp', true_comp)
                    print()
    print('Number of prediction', num_predicted)
    print('Correct composition', num_correct_comp)
    print('Incorrect_comp_scan', incorrect_comp_scan)


if __name__ == '__main__':
    main(parse_args())

