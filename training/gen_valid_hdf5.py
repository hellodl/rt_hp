import argparse
import os
import cv2
import json
import numpy as np
from scipy.spatial.distance import cdist
import h5py
import matplotlib.pyplot as plt
import pylab
import struct
import logging

dataset_dir = None
tr_ann_path = None
tr_img_dir = None
val_ann_path = None
val_img_dir = None
datasets = []
joint_all = []

tr_hdf5_path = None
val_hdf5_path = None
val_size = None
joint_num = None
tr_mask_dir = ''
val_mask_dir = ''


def float2bytes(floats):
    if type(floats) is float:
        floats = [floats]
    if type(floats) is int:
        floats = [float(floats)]

    if type(floats) is list and len(floats) > 0 and type(floats[0]) is list:
        floats = floats[0]

    return struct.pack('%sf' % len(floats), *floats)


def argsParse():
    global dataset_dir, tr_ann_path, tr_img_dir, val_ann_path, val_img_dir
    global datasets, tr_hdf5_path, val_hdf5_path, val_size, joint_num, joint_all

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir')
    parser.add_argument('--train_ann_dir')
    parser.add_argument('--train_img_dir')
    parser.add_argument('--valid_ann_dir')
    parser.add_argument('--valid_img_dir')
    parser.add_argument('--train_hdf5_path')
    parser.add_argument('--valid_hdf5_path')
    parser.add_argument('--valid_size')
    parser.add_argument('--joint_num')
    parser.add_argument('--log_path')

    args = parser.parse_args()

    dataset_dir = args.data_dir
    tr_ann_path = os.path.join(dataset_dir, args.train_ann_dir)
    tr_img_dir = os.path.join(dataset_dir, args.train_img_dir)

    val_ann_path = os.path.join(dataset_dir, args.valid_ann_dir)
    val_img_dir = os.path.join(dataset_dir, args.valid_img_dir)

    datasets = [
        (val_ann_path, val_img_dir, "AIC"),
        #(tr_ann_path, tr_img_dir, "AIC")
    ]

    tr_hdf5_path = os.path.join(dataset_dir, args.train_hdf5_path)
    val_hdf5_path = os.path.join(dataset_dir, args.valid_hdf5_path)
    val_size = int(args.valid_size)
    joint_num = int(args.joint_num)
    log_filepath = os.path.join(dataset_dir, args.log_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y',
                        filename=log_filepath,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # print('data_dir: ', dataset_dir)
    # print('train_ann_dir: ', tr_ann_path)
    # print('train_img_dir: ', tr_img_dir)
    # print('valid_ann_dir: ', val_ann_path)
    # print('valid_img_dir: ', val_img_dir)
    print('train_hdf5_path: ', tr_hdf5_path)
    print('valid_hdf5_path: ', val_hdf5_path)


def argsCheck():
    pass


def process():
    for ds in datasets:  # traverse dataset including 'train' and 'valid'
        count = 0
        ann_path = ds[0]
        img_path = ds[1]
        dataset_type = ds[2]

        # print(ann_path)
        with open(ann_path, 'r') as json_object:
            ky_annotations = json.load(json_object)

        dataset_size = len(ky_annotations)
        for i, img in enumerate(ky_annotations):  # traverse all img
            if count % 500 == 0:
                print("%s/%s annotations have been analyzied." % (count, dataset_size))

            img_id = img['image_id']
            kp_anns = img['keypoint_annotations']
            bb_anns = img['human_annotations']

            # print(img_path+ '/' + img_id + '.jpg')
            image = cv2.imread(img_path + '/' + img_id + '.jpg')
            h, w, c = image.shape

            persons = []
            # prev_center = []
            skip_cnt = 0
            for p in kp_anns.keys():  # traverse all people in one img
                p_ann = kp_anns[p]
                bbox = bb_anns[p]
                bbox_w = bbox[2] - bbox[0] + 1
                bbox_h = bbox[3] - bbox[1] + 1

                area = bbox_w * bbox_h
                num_keypoints = len(p_ann)//3 - p_ann[2::3].count(3)

                # skip this person if parts number is too low or if
                # segmentation area is too small
                if area < 32 * 32:
                    skip_cnt += 1
                    logging.warning("%s has been ignored for been too small." % p)
                    continue
                elif num_keypoints < 4:
                    skip_cnt += 1
                    logging.warning("%s has been ignored for not enough keypoints." % p)
                    continue

                pers = dict()
                person_center = [(bbox[0] + bbox[2]) // 2,
                                 (bbox[1] + bbox[3]) // 2]

                # skip this person if the distance to exiting person is too small
                """
               ignore this skipping method...  edited by g.hy

               flag = 0
               for pc in prev_center:
                   a = np.expand_dims(pc[:2], axis=0)
                   b = np.expand_dims(person_center, axis=0)
                   dist = cdist(a, b)[0]
                   if dist < pc[2]*0.3:
                       flag = 1
                       continue

               if flag == 1:
                   continue
               """

                pers["objpos"] = person_center
                pers["bbox"] = bbox
                pers["segment_area"] = area
                pers["num_keypoints"] = num_keypoints

                pers["joint"] = np.zeros((joint_num, 3))
                for part in range(joint_num):
                    pers["joint"][part, 0] = p_ann[part * 3]
                    pers["joint"][part, 1] = p_ann[part * 3 + 1]

                    # print(pers["joint"][part, 1])

                    if p_ann[part * 3 + 2] == 1:
                        pers["joint"][part, 2] = 1  # visible
                    elif p_ann[part * 3 + 2] == 2:
                        pers["joint"][part, 2] = 0  # labeled but not visible
                    else:
                        pers["joint"][part, 2] = 2

                pers["scale_provided"] = max((bbox[2] - bbox[0] + 1) / 368, (bbox[3] - bbox[1] + 1) / 368)

                persons.append(pers)
                # prev_center.append(np.append(person_center, max(bbox[2], bbox[3])))

            if skip_cnt > 0:
                logging.warning("%s/%s people ignored. (img: %s ...)" % (skip_cnt, len(kp_anns), img_id[0:8]))

            if len(persons) > 0:
                joint_all.append(dict())

                joint_all[count]["dataset"] = dataset_type
                if count < val_size:
                    isValidation = 1
                else:
                    isValidation = 0

                joint_all[count]["isValidation"] = isValidation

                joint_all[count]["img_width"] = w
                joint_all[count]["img_height"] = h
                joint_all[count]["image_id"] = img_id
                joint_all[count]["annolist_index"] = i

                # set image path
                joint_all[count]["img_paths"] = os.path.join(img_path, '%s.jpg' % img_id)
                joint_all[count]["mask_miss_paths"] = 'do not care'
                joint_all[count]["mask_all_paths"] = 'do not care'

                # set the main person
                joint_all[count]["objpos"] = persons[0]["objpos"]
                joint_all[count]["bbox"] = persons[0]["bbox"]
                joint_all[count]["segment_area"] = persons[0]["segment_area"]
                joint_all[count]["num_keypoints"] = persons[0]["num_keypoints"]
                joint_all[count]["joint_self"] = persons[0]["joint"]
                # print(joint_all[count]["joint_self"])
                joint_all[count]["scale_provided"] = persons[0]["scale_provided"]

                # set other persons
                joint_all[count]["joint_others"] = []
                joint_all[count]["scale_provided_other"] = []
                joint_all[count]["objpos_other"] = []
                joint_all[count]["bbox_other"] = []
                joint_all[count]["segment_area_other"] = []
                joint_all[count]["num_keypoints_other"] = []

                for ot in range(1, len(persons)):
                    joint_all[count]["joint_others"].append(persons[ot]["joint"])
                    joint_all[count]["scale_provided_other"].append(persons[ot]["scale_provided"])
                    joint_all[count]["objpos_other"].append(persons[ot]["objpos"])
                    joint_all[count]["bbox_other"].append(persons[ot]["bbox"])
                    joint_all[count]["segment_area_other"].append(persons[ot]["segment_area"])
                    joint_all[count]["num_keypoints_other"].append(persons[ot]["num_keypoints"])

                joint_all[count]["people_index"] = 0
                lenOthers = len(persons) - 1

                joint_all[count]["numOtherPeople"] = lenOthers
                count += 1


def writeHDF5():
    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("datum")
    tr_write_count = 0

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("datum")
    val_write_count = 0

    data = joint_all
    numSample = len(data)

    isValidationArray = [data[i]['isValidation'] for i in range(numSample)]
    val_total_write_count = isValidationArray.count(1.0)
    tr_total_write_count = len(data) - val_total_write_count

    random_order = [i for i, el in enumerate(range(len(data)))]

    for count in range(numSample):
        idx = random_order[count]

        img = cv2.imread(data[idx]['img_paths'])
        # mask_all = cv2.imread(data[idx]['mask_all_paths'], 0)
        # mask_miss = cv2.imread(data[idx]['mask_miss_paths'], 0)

        isValidation = data[idx]['isValidation']

        height = img.shape[0]
        width = img.shape[1]

        if width < 64:
            h_border_width = 64 - width
            # cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)
            img = cv2.copyMakeBorder(img, 0, 0, 0, h_border_width, cv2.BORDER_CONSTANT,
                                     value=(128, 128, 128))

            print('saving padded image!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            cv2.imwrite('padded_img.jpg', img)
            width = 64

        meta_data = np.zeros(shape=(height, width, 1), dtype=np.uint8)

        clidx = 0  # current line index

        for i in range(len(data[idx]['dataset'])):
            meta_data[clidx][i] = ord(data[idx]['dataset'][i])
        clidx = clidx + 1

        # image height, image width
        height_binary = float2bytes(data[idx]['img_height'])
        for i in range(len(height_binary)):
            meta_data[clidx][i] = height_binary[i]
        width_binary = float2bytes(data[idx]['img_width'])
        for i in range(len(width_binary)):
            meta_data[clidx][4 + i] = width_binary[i]

        clidx = clidx + 1
        # (a) isValidation(uint8), numOtherPeople (uint8), people_index (uint8),
        # annolist_index (float), writeCount(float), totalWriteCount(float)
        meta_data[clidx][0] = data[idx]['isValidation']
        meta_data[clidx][1] = data[idx]['numOtherPeople']
        meta_data[clidx][2] = data[idx]['people_index']
        annolist_index_binary = float2bytes(data[idx]['annolist_index'])
        for i in range(len(annolist_index_binary)):  # 3,4,5,6
            meta_data[clidx][3 + i] = annolist_index_binary[i]

        if isValidation:
            count_binary = float2bytes(float(val_write_count))
        else:
            count_binary = float2bytes(float(tr_write_count))
        for i in range(len(count_binary)):
            meta_data[clidx][7 + i] = count_binary[i]

        if isValidation:
            totalWriteCount_binary = float2bytes(float(val_total_write_count))
        else:
            totalWriteCount_binary = float2bytes(float(tr_total_write_count))
        for i in range(len(totalWriteCount_binary)):
            meta_data[clidx][11 + i] = totalWriteCount_binary[i]

        nop = int(data[idx]['numOtherPeople'])

        # (b) objpos_x (float), objpos_y (float)
        clidx = clidx + 1
        objpos_binary = float2bytes(data[idx]['objpos'])  # main people

        for i in range(len(objpos_binary)):
            meta_data[clidx][i] = objpos_binary[i]

        # (c) scale_provided (float)
        clidx = clidx + 1
        scale_provided_binary = float2bytes(data[idx]['scale_provided'])
        for i in range(len(scale_provided_binary)):
            meta_data[clidx][i] = scale_provided_binary[i]

        # (d) joint_self (3*16) (float) (3 line)
        clidx = clidx + 1
        joints = np.asarray(data[idx]['joint_self']).T.tolist()  # transpose to 3*16
        for i in range(len(joints)):
            row_binary = float2bytes(joints[i])
            for j in range(len(row_binary)):
                meta_data[clidx][j] = row_binary[j]

            clidx = clidx + 1
        # (e) check nop, prepare arrays
        if (nop != 0):
            joint_other = data[idx]['joint_others']
            objpos_other = data[idx]['objpos_other']
            scale_provided_other = data[idx]['scale_provided_other']
            # (f) objpos_other_x (float), objpos_other_y (float) (nop lines)
            for i in range(nop):
                objpos_binary = float2bytes(objpos_other[i])
                for j in range(len(objpos_binary)):
                    meta_data[clidx][j] = objpos_binary[j]
                clidx = clidx + 1
            # (g) scale_provided_other (nop floats in 1 line)
            scale_provided_other_binary = float2bytes(scale_provided_other)
            for j in range(len(scale_provided_other_binary)):
                meta_data[clidx][j] = scale_provided_other_binary[j]
            clidx = clidx + 1
            # (h) joint_others (3*16) (float) (nop*3 lines)
            for n in range(nop):
                joints = np.asarray(joint_other[n]).T.tolist()  # transpose to 3*16
                for i in range(len(joints)):
                    row_binary = float2bytes(joints[i])
                    for j in range(len(row_binary)):
                        meta_data[clidx][j] = row_binary[j]
                    clidx = clidx + 1

        img4ch = np.concatenate((img, meta_data), axis=2)
        img4ch = np.transpose(img4ch, (2, 0, 1))

        if isValidation:
            key = '%07d' % val_write_count
            val_grp.create_dataset(key, data=img4ch, chunks=None)
            val_write_count += 1
        else:
            key = '%07d' % tr_write_count
            tr_grp.create_dataset(key, data=img4ch, chunks=None)
            tr_write_count += 1

        print('Writing sample %d/%d' % (count+1, numSample))
    print('trn_size:', tr_total_write_count, ' val_size: ', val_total_write_count)


if __name__ == '__main__':
    argsParse()
    argsCheck()
    process()
    writeHDF5()
