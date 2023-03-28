import os
from pathlib import Path
from tqdm import tqdm
import argparse
import random
import numpy as np
random.seed(666)
np.random.seed(666)

def generate_list(source_file, save_file):
    save_file = open(save_file, 'w')
    idx = 0
    with tqdm(total=5179511) as pbar:
        pbar.set_description('Processing:')
        with open(source_file, 'r') as f:
            for line in f.readlines():
                idx_label = str(idx)+" "+line.split("\t")[-1]
                idx = idx + 1
                save_file.writelines(idx_label)
                pbar.update(1)

    save_file.close()
    print("idx_label list are built.")


def split_ms1mv3_by_open_class(input_file, classes, ratio=0.3, dataset='ms1mv3'):
    input_file = Path(input_file)
    bucket = [[] for _ in range(classes)]
    random.seed(666)
    np.random.seed(666)
    all_class_list = np.arange(0, classes)
    np.random.shuffle(all_class_list)
    old_class_list = all_class_list[:int(classes * ratio)]
    old_class_bool_index = np.array([False for _ in range(classes)])
    old_class_bool_index[old_class_list] = True

    with open(input_file, 'r') as f:
        for line in f.readlines():
            bucket[int(line.split(" ")[-1])].append(line)

    old_class_file = open(
        input_file.parent / f"{dataset}_train_old_{int(ratio * 100)}percent_class_openclass.txt", "w")
    new_class_file = open(
        input_file.parent / f"{dataset}_train_new_{int((1-ratio) * 100)}percent_class_openclass.txt", "w")

    old_class_num, old_img_num = 0, 0
    new_class_num, new_img_num = 0, 0

    for i in range(classes):
        if old_class_bool_index[i]:
            for j in bucket[i]:
                old_class_file.write(j.replace('\n', ' ')+str(old_class_num)+'\n')
                old_img_num += 1
            old_class_num += 1
        else:
            for j in bucket[i]:
                new_class_file.write(j.replace('\n', ' ')+str(new_class_num)+'\n')
                new_img_num += 1
            new_class_num += 1
    old_class_file.close()
    new_class_file.close()
    print("Old class count: %d, img count: %d" % (old_class_num, old_img_num))
    print("New class count: %d, img count: %d" % (new_class_num, new_img_num))
    print("Done.")


def split_ms1mv3_by_open_data(input_file, classes, ratio=0.3, dataset='ms1mv3'):
    input_file = Path(input_file)
    bucket = np.zeros([classes], dtype=int)
    paths = [[] for _ in range(classes)]
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = int(line_splits[-1]) if line_splits[-1] != '\n' else int(line_splits[-2])
            bucket[current_label] += 1
            paths[current_label].append(line)

    old_training_data_file = open(
        input_file.parent/f"{dataset}_train_old_{int(ratio * 100)}percent_opendata.txt" , "w")

    new_training_data_file = open(
        input_file.parent / f"{dataset}_train_new_{int((1-ratio) * 100)}percent_opendata.txt", "w")

    old_num = 0
    new_num = 0
    for i in range(classes):
        if bucket[i] == 0:
            continue
        curr_count = bucket[i]
        if curr_count == 1:
            old_training_data_file.write(paths[i][0])
            new_training_data_file.write(paths[i][0])
            old_num += 1
            new_num += 1
        elif curr_count>1 and curr_count < 4:
            old_training_data_file.write(paths[i][0])
            old_num += 1
            for j in range(1, curr_count):
                new_training_data_file.write(paths[i][j])
                new_num += 1
        else:
            random.seed(666 + i)
            np.random.seed(666 + i)
            all_list = np.arange(0, curr_count)
            np.random.shuffle(all_list)
            old_list = all_list[:int(curr_count * ratio)]
            old_bool_index = np.array([False for _ in range(curr_count)])
            old_bool_index[old_list] = True
            for j, element in enumerate(paths[i]):
                if old_bool_index[j]:
                    old_training_data_file.write(element)
                    old_num += 1
                else:
                    new_training_data_file.write(element)
                    new_num += 1

    old_training_data_file.close()
    new_training_data_file.close()
    print("Old data count: %d" % old_num)
    print("New data count: %d" % new_num)
    print("Done.")


def split_ms1mv3_by_extended_class(input_file, classes, ratio=0.3, dataset='ms1mv3'):
    input_file = Path(input_file)
    bucket = [[] for _ in range(classes)]
    random.seed(666)
    np.random.seed(666)
    all_class_list = np.arange(0, classes)
    np.random.shuffle(all_class_list)
    old_class_list = all_class_list[:int(classes * ratio)]
    old_class_bool_index = np.array([False for _ in range(classes)])
    old_class_bool_index[old_class_list] = True

    with open(input_file, 'r') as f:
        for line in f.readlines():
            bucket[int(line.split(" ")[-1])].append(line)

    old_class_file = open(
        input_file.parent / f"{dataset}_train_old_{int(ratio * 100)}percent_class_extendedclass.txt", "w")
    new_class_file = open(
        input_file.parent / f"{dataset}_train_new_{int(100)}percent_class_extendedclass.txt", "w")

    old_class_num, old_img_num = 0, 0
    new_class_num, new_img_num = 0, 0

    for i in range(classes):
        if old_class_bool_index[i]:
            for j in bucket[i]:
                old_class_file.write(j.replace('\n', ' ')+str(old_class_num)+'\n')
                old_img_num += 1
            old_class_num += 1
        for j in bucket[i]:
            new_class_file.write(j.replace('\n', ' ')+str(new_class_num)+'\n')
            new_img_num += 1
        new_class_num += 1
    old_class_file.close()
    new_class_file.close()
    print("Old class count: %d, img count: %d" % (old_class_num, old_img_num))
    print("New class count: %d, img count: %d" % (new_class_num, new_img_num))
    print("Done.")


def split_ms1mv3_by_extended_data(input_file, classes, ratio=0.3, dataset='ms1mv3'):
    input_file = Path(input_file)
    bucket = np.zeros([classes], dtype=int)
    paths = [[] for _ in range(classes)]
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = int(line_splits[-1]) if line_splits[-1] != '\n' else int(line_splits[-2])
            bucket[current_label] += 1
            paths[current_label].append(line)

    old_training_data_file = open(
        input_file.parent/f"{dataset}_train_old_{int(ratio * 100)}percent_extendeddata.txt" , "w")

    new_training_data_file = open(
        input_file.parent / f"{dataset}_train_new_{int(100)}percent_extendeddata.txt", "w")

    old_num = 0
    new_num = 0
    for i in range(classes):
        if bucket[i] == 0:
            continue
        curr_count = bucket[i]
        if curr_count == 1:
            old_training_data_file.write(paths[i][0])
            new_training_data_file.write(paths[i][0])
            old_num += 1
            new_num += 1
        elif curr_count>1 and curr_count < 4:
            old_training_data_file.write(paths[i][0])
            old_num += 1
            for j in range(0, curr_count):
                new_training_data_file.write(paths[i][j])
                new_num += 1
        else:
            random.seed(666 + i)
            np.random.seed(666 + i)
            all_list = np.arange(0, curr_count)
            np.random.shuffle(all_list)
            old_list = all_list[:int(curr_count * ratio)]
            old_bool_index = np.array([False for _ in range(curr_count)])
            old_bool_index[old_list] = True
            for j, element in enumerate(paths[i]):
                if old_bool_index[j]:
                    old_training_data_file.write(element)
                    old_num += 1
                new_training_data_file.write(element)
                new_num += 1

    old_training_data_file.close()
    new_training_data_file.close()
    print("Old data count: %d" % old_num)
    print("New data count: %d" % new_num)
    print("Done.")


def split_ms1mv3_by_identical_data(input_file, classes, ratio=0.3, dataset='ms1mv3'):
    input_file = Path(input_file)
    bucket = np.zeros([classes], dtype=int)
    paths = [[] for _ in range(classes)]
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = int(line_splits[-1]) if line_splits[-1] != '\n' else int(line_splits[-2])
            bucket[current_label] += 1
            paths[current_label].append(line)

    identical_training_data_file = open(
        input_file.parent/f"{dataset}_train_identical_{int(ratio * 100)}percent_data.txt" , "w")

    identical_num = 0
    for i in range(classes):
        if bucket[i] == 0:
            continue
        curr_count = bucket[i]
        if curr_count == 1:
            identical_training_data_file.write(paths[i][0])
            identical_num += 1
        elif curr_count>1 and curr_count < 4:
            identical_training_data_file.write(paths[i][0])
            identical_num += 1
        else:
            random.seed(666 + i)
            np.random.seed(666 + i)
            all_list = np.arange(0, curr_count)
            np.random.shuffle(all_list)
            identical_list = all_list[:int(curr_count * ratio)]
            identical_bool_index = np.array([False for _ in range(curr_count)])
            identical_bool_index[identical_list] = True
            for j, element in enumerate(paths[i]):
                if identical_bool_index[j]:
                    identical_training_data_file.write(element)
                    identical_num += 1

    identical_training_data_file.close()
    print("identical data count: %d" % identical_num)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', default='/ProjectRoot/webpage_pretrain/runner_jlt/dataset/ms1m-retinaface-t1/ms1m-retinaface-t1', type=str, help='')
    parser.add_argument('--split_file', default='/ProjectRoot/webpage_pretrain/runner_jlt/dataset/ms1m-retinaface-t1/ms1m-retinaface-t1/label_93431.txt', type=str, help='')
    parser.add_argument('--processlist_file', action='store_true')
    parser.add_argument('--split_ms1mv3_by_extended_data', action='store_true')
    parser.add_argument('--split_ms1mv3_by_extended_class', action='store_true')
    parser.add_argument('--split_ms1mv3_by_open_data', action='store_true')
    parser.add_argument('--split_ms1mv3_by_open_class', action='store_true')
    parser.add_argument('--split_ms1mv3_by_identical_data', action='store_true')
    parser.add_argument('--split_ratio', default=0.3, type=float, help='')
    args = parser.parse_args()

    root_path = args.root_path
    Path(root_path).mkdir(parents=True, exist_ok=True)

    if args.processlist_file:
        source_file = os.path.join(root_path, 'train.lst')
        save_file = os.path.join(root_path, 'label_93431.txt')
        generate_list(source_file, save_file)

    cls_num_dic = {'gldv2': 81313, 'imagenet': 1000, 'places365': 365, 'market': 1502, 'ms1mv3': 93431}
    if args.split_ms1mv3_by_open_data:
        whole_training_file = os.path.join(root_path, 'label_93431.txt')
        assert os.path.isfile(whole_training_file), \
            "Please download label_93431.txt first."
        split_ms1mv3_by_open_data(whole_training_file, cls_num_dic['ms1mv3'], args.split_ratio, dataset='ms1mv3')

    if args.split_ms1mv3_by_open_class:
        whole_training_file = os.path.join(root_path, 'label_93431.txt')
        assert os.path.isfile(whole_training_file), \
            "Please download label_93431.txt first."
        split_ms1mv3_by_open_class(whole_training_file, cls_num_dic['ms1mv3'], args.split_ratio, dataset='ms1mv3')

    if args.split_ms1mv3_by_identical_data:
        whole_training_file = os.path.join(root_path, 'label_93431.txt')
        assert os.path.isfile(whole_training_file), \
            "Please download label_93431.txt first."
        split_ms1mv3_by_identical_data(whole_training_file, cls_num_dic['ms1mv3'], args.split_ratio, dataset='ms1mv3')

    if args.split_ms1mv3_by_extended_data:
        whole_training_file = os.path.join(root_path, 'label_93431.txt')
        assert os.path.isfile(whole_training_file), \
            "Please download label_93431.txt first."
        split_ms1mv3_by_extended_data(whole_training_file, cls_num_dic['ms1mv3'], args.split_ratio, dataset='ms1mv3')

    if args.split_ms1mv3_by_extended_class:
        whole_training_file = os.path.join(root_path, 'label_93431.txt')
        assert os.path.isfile(whole_training_file), \
            "Please download label_93431.txt first."
        split_ms1mv3_by_extended_class(whole_training_file, cls_num_dic['ms1mv3'], args.split_ratio, dataset='ms1mv3')