"""
Shapenet Core v1, Chairs, fine-grained classification;
prepare for train and test;
1: split different fine-grained classes
2: train/test split
3: points generation

conda env: tf_1_10
"""
import os
import sys
import glob
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import pylab as pl
import numpy as np
import argparse
from collections import OrderedDict

from ocnn.virtualscanner import VirtualScanner
from ocnn.octree import Points
from FarthestPointSampling.tf_sampling import FPS
import pdb


##############################################################
# arguments settings
##############################################################
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='statistic', help="set the running mode of this script")
parser.add_argument('--synoffset', type=str, default='03001627', help="set the synoffset from ShapeNetCore.v1 to process")
parser.add_argument('--test_num', type=int, default=50, help="set the instance number of the test pile")
parser.add_argument('--sample_num', type=int, default=4096, help="set the sampling number of the 3D point clouds")
parser.add_argument('--write', action='store_true', help="default is False, unless explicitly type --write")

FLAGS = parser.parse_args()

##############################################################
# basic functions
##############################################################

def str2set(lemmas):
    """
    lemmas: a string, contains multiple synnames of a object
        'ball chair, cantilever chair, armchari, chair'
    """
    return set(lemmas.lower().split(','))


def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()


def dataset_statistics(synname, synlabeln, syn_dict):
    syn_dict_sorted = OrderedDict(sorted(syn_dict.items(), key=lambda t: t[1], reverse=True))

    #######################################################
    # confusion matrix
    syn_name_tab = [k for k in syn_dict_sorted.keys()]
    cm = np.zeros((len(syn_name_tab), len(syn_name_tab)))
    for i in range(len(syn_name_tab)):
        ref_name = syn_name_tab[i]
        for j in range(len(synname)):
            cur_set = synname[j]
            if ref_name in cur_set:
                idx = [syn_name_tab.index(name) for name in cur_set]
                cm[i, idx] += 1

    for r in range(len(cm)):
        cm[r] = cm[r] / cm[r, r]

    cm = (cm + np.transpose(cm)) / 2

    plot_confusion_matrix(cm, syn_name_tab)

    plot_confusion_matrix(cm[:10, :10], syn_name_tab[:10])


    #######################################################
    # number of fine-grained subsets
    plt.figure(figsize=(10, 10))
    plt.bar(syn_dict_sorted.keys(), syn_dict_sorted.values())
    plt.xticks(rotation=75)
    plt.show()

    labeln = np.array(synlabeln)  # the number of multi-label models
    numtab = np.unique(labeln)
    numbers = []
    for n in numtab:
        # the model number, of which has n labels
        numbers.append(np.sum(labeln == n))

    plt.figure(figsize=(9, 6))
    plt.bar(numtab, numbers)
    plt.show()


def gen_split_files(data_dict, labels, save_root, test_pile):
    for l in labels:
        train_fn = 'train_{:s}.txt'.format(l)
        test_fn = 'test_{:s}.txt'.format(l)
        idx = np.arange(len(data_dict[l]), dtype=np.int32)
        np.random.shuffle(idx)

        with open(os.path.join(save_root, train_fn), 'w+') as f:
            for i in range(len(idx) - test_pile):
                st = data_dict[l][idx[i]].split('.')[-1]
                f.write(st+'\n')

        with open(os.path.join(save_root, test_fn), 'w+') as f:
            for i in range(len(idx) - test_pile, len(idx)):
                st = data_dict[l][idx[i]].split('.')[-1]
                f.write(st+'\n')


def load_file_list(split_dir):
    # load file list from split txt
    if not os.path.exists(split_dir):
        print('No such file! {:s}'.format(split_dir))
        raise RuntimeError
    with open(split_dir, 'r') as f:
        file_list = f.readlines()
    return [file.strip('\n') for file in file_list]


def getSubdirectories(base):
    return [folder for folder in os.listdir(base) if os.path.isdir(os.path.join(base, folder))]


def mkdirs_img(save_root, labels):
    img_root = os.path.join(save_root, 'imgs')
    if not os.path.exists(img_root):
        os.mkdir(img_root)
    for l in labels:
        folder_n = 'test_{:s}'.format(l)
        folder_dir = os.path.join(img_root, folder_n)
        os.mkdir(folder_dir) if not os.path.exists(folder_dir) else print('already exist {:s}'.format(folder_dir))


#########################
# Utils for data prepare
#########################

def parse_csv(synoffset):
    csvfile = os.path.join('../data/ShapeNetCore.v1', synoffset+'.csv')
    syndf = pd.read_csv(csvfile, encoding='utf-8')
    modelid = []
    synname = []
    synbag = set()
    for i in range(syndf.index.stop):
        lemmas = str2set(syndf.loc[i].wnlemmas)
        synname.append(lemmas)
        synbag = synbag.union(lemmas)
        modelid.append(syndf.loc[i].fullId)

    syn_dict = {syn:0 for syn in synbag}

    synlabeln = []  # syn label numbers
    for lemmas in synname:
        synlabeln.append(len(lemmas))
        for l in lemmas:
            syn_dict[l] += 1
    return synname, synlabeln, syn_dict, modelid


def setup_subsets_according_to_statistics(synoffset, synname, modelid):
    labels = None
    subset_ndict = None

    if synoffset == '03001627':
        # Chair
        labels = ['A', 'B', 'C', 'D', 'E']
        sub_synset = {labels[0]: ['straight chair', 'side chair'],
                      labels[1]: ['armchair'],
                      labels[2]: ['club chair'],
                      labels[3]: ['swivel chair'],
                      labels[4]: ['overstuffed chair', 'lounge chair', 'easy chair']}
        subset_ndict = {l: set() for l in labels}
        for i in range(len(labels)):
            cur_set = set(sub_synset[labels[i]])
            for j in range(len(synname)):
                tmp_set = synname[j]
                common = cur_set.intersection(tmp_set)
                if len(common) > 0:
                    subset_ndict[labels[i]].add(modelid[j])
        # erase the intersected model ids
        # remove D E from A, remove C D E from B, remove D E from C, remove E from D
        subset_ndict['A'] -= subset_ndict['A'].intersection(
            subset_ndict['B'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['B'] -= subset_ndict['B'].intersection(
            subset_ndict['C'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['C'] -= subset_ndict['C'].intersection(subset_ndict['D'].union(subset_ndict['E']))
        subset_ndict['D'] -= subset_ndict['D'].intersection(subset_ndict['E'])
    elif synoffset == '02691156':
        # Airplane
        labels = ['A', 'B', 'C', 'D', 'E']
        sub_synset = {labels[0]: ['airliner'],
                      labels[1]: ['jet', 'jet-propelled plane', 'jet plane'],
                      labels[2]: ['fighter', 'attack aircraft', 'fighter aircraft'],
                      labels[3]: ['bomber'],
                      labels[4]: ['propeller plane', 'straight wing']}
        subset_ndict = {l: set() for l in labels}
        for i in range(len(labels)):
            cur_set = set(sub_synset[labels[i]])
            for j in range(len(synname)):
                tmp_set = synname[j]
                common = cur_set.intersection(tmp_set)
                if len(common) > 0:
                    subset_ndict[labels[i]].add(modelid[j])
        # erase the intersected model ids
        # remove D E from A, remove C D E from B, remove D E from C, remove E from D
        subset_ndict['A'] -= subset_ndict['A'].intersection(
            subset_ndict['B'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['B'] -= subset_ndict['B'].intersection(
            subset_ndict['C'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['C'] -= subset_ndict['C'].intersection(subset_ndict['D'].union(subset_ndict['E']))
        subset_ndict['D'] -= subset_ndict['D'].intersection(subset_ndict['E'])
    elif synoffset == '02958343':
        # Car
        labels = ['A', 'B', 'C', 'D', 'E']
        sub_synset = {labels[0]: ['coupe'],
                      labels[1]: ['sedan', 'saloon'],
                      labels[2]: ['s.u.v', 'suv', 'sport utility vehicle', 'sport utility'],
                      labels[3]: ['racer', 'racing car', 'race car'],
                      labels[4]: ['convertible']}
        subset_ndict = {l: set() for l in labels}
        for i in range(len(labels)):
            cur_set = set(sub_synset[labels[i]])
            for j in range(len(synname)):
                tmp_set = synname[j]
                common = cur_set.intersection(tmp_set)
                if len(common) > 0:
                    subset_ndict[labels[i]].add(modelid[j])
        # erase the intersected model ids
        # remove D E from A, remove C D E from B, remove D E from C, remove E from D
        subset_ndict['A'] -= subset_ndict['A'].intersection(
            subset_ndict['B'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['B'] -= subset_ndict['B'].intersection(
            subset_ndict['C'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['C'] -= subset_ndict['C'].intersection(subset_ndict['D'].union(subset_ndict['E']))
        subset_ndict['D'] -= subset_ndict['D'].intersection(subset_ndict['E'])
    elif synoffset == '04379243':
        # Table
        labels = ['A', 'B', 'C', 'D', 'E']
        sub_synset = {labels[0]: ['desk'],
                      labels[1]: ['cocktail table', 'coffee table'],
                      labels[2]: ['rectangular table'],
                      labels[3]: ['workshop table'],
                      labels[4]: ['console table', 'console']}
        subset_ndict = {l: set() for l in labels}
        for i in range(len(labels)):
            cur_set = set(sub_synset[labels[i]])
            for j in range(len(synname)):
                tmp_set = synname[j]
                common = cur_set.intersection(tmp_set)
                if len(common) > 0:
                    subset_ndict[labels[i]].add(modelid[j])
        # erase the intersected model ids
        # remove D E from A, remove C D E from B, remove D E from C, remove E from D
        subset_ndict['A'] -= subset_ndict['A'].intersection(
            subset_ndict['B'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['B'] -= subset_ndict['B'].intersection(
            subset_ndict['C'].union(subset_ndict['D'], subset_ndict['E']))
        subset_ndict['C'] -= subset_ndict['C'].intersection(subset_ndict['D'].union(subset_ndict['E']))
        subset_ndict['D'] -= subset_ndict['D'].intersection(subset_ndict['E'])

    print('subset elements num:', [len(subset_ndict[l]) for l in labels])
    # [1967, 1616, 778, 408, 394]-->[1854, 1090, 655, 389, 394]
    # pdb.set_trace()
    print([[len(subset_ndict[l_ref].intersection(subset_ndict[l])) for l in labels] for l_ref in labels])

    return labels, subset_ndict


def gen_split_warper(synoffset, subset_ndict, labels, test_num=50, write_flag=False):
    data_dict = {l:list(subset_ndict[l]) for l in labels}
    save_root = os.path.join('../data/shapenet', synoffset)
    test_pile = test_num  # test shape number for each subset
    if write_flag:
        gen_split_files(data_dict, labels, save_root, test_pile)
    mkdirs_img(save_root, labels)


def load_splits(synoffset):
    split_root = os.path.join('../data/shapenet', synoffset)
    fnames = ['test_A', 'train_A', 'test_B', 'train_B',
              'test_C', 'train_C', 'test_D', 'train_D', 'test_E', 'train_E']
    fn_list = []
    for fn in fnames:
        f_dir = os.path.join(split_root, fn+'.txt')
        with open(f_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                fn_list.append(line.strip())
    return fn_list


if __name__ == '__main__':
    synoffset = FLAGS.synoffset

    """ parse the csv file """
    synname, synlabeln, syn_dict, modelid = parse_csv(synoffset)

    if FLAGS.mode == "statistic":
        print('-'*30)
        print('| Mode: ', FLAGS.mode)
        print('| synoffset: ', FLAGS.synoffset)
        print('-'*30)
        """ get the statistics """
        dataset_statistics(synname, synlabeln, syn_dict)
    elif FLAGS.mode == "gen_split":
        print('-' * 30)
        print('| Mode: ', FLAGS.mode)
        print('| synoffset: ', FLAGS.synoffset)
        print('-' * 30)
        """ subset selection for transfer """
        labels, subset_ndict = setup_subsets_according_to_statistics(synoffset, synname, modelid)
        """ train/test split, save file lists into .txt """
        gen_split_warper(synoffset, subset_ndict, labels, test_num=FLAGS.test_num, write_flag=FLAGS.write)
    elif FLAGS.mode == "sampling":
        print('-' * 30)
        print('| Mode: ', FLAGS.mode)
        print('| synoffset: ', FLAGS.synoffset)
        print('-' * 30)
        """ sampling points, all chairs/other classes """
        data_root = os.path.join('../data/ShapeNetCore.v1', synoffset)
        model_names = getSubdirectories(data_root)
        # pdb.set_trace()
        snum = FLAGS.sample_num  # 4096 # 2048  # sampling num
        out_root = os.path.join('../data/shapenet', synoffset, 'points')
        if not os.path.exists(out_root):
            os.mkdir(out_root)
        save_root = os.path.join(out_root, 'hdf5_' + str(snum))
        if not os.path.exists(save_root):
            os.mkdir(save_root)

        model_names_in_split = load_splits(synoffset)
        assert sum([(m in model_names) for m in model_names_in_split]) == len(model_names_in_split), "Error!! Invalid model names in split file!"
        count = 0

        for fn in model_names_in_split:
            save_fn = os.path.join(save_root, fn)
            print('...............{:d} of {:d}'.format(count+1, len(model_names_in_split)))
            print('model name {:s}'.format(fn))
            if os.path.exists(save_fn+'.hd5'):
                print('hdf5 file {:s} already exists... continue!'.format(fn))
                count += 1
                continue
            if not os.path.exists(os.path.join(out_root, fn + '.points')):
                scanner = VirtualScanner(filepath=os.path.join(data_root, fn, 'model.obj'),
                                         view_num=6, flags=False, normalize=True)
                scanner.save(os.path.join(out_root, fn + '.points'))
            else:
                pass

            # read .points file and parse the point data
            points = Points(os.path.join(out_root, fn + '.points'))
            [pts, norms] = points.get_points_data()

            # Farthest Point Sampling

            print('sampling...')

            sampled_pts = FPS(pts, snum, save_fn)
            pts_new = np.squeeze(sampled_pts)
            count += 1
