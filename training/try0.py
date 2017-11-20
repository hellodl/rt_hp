import cv2
import json
import numpy as np
import os

# to do list
# h, w
# numPeople  # in each image
# img_id
# bounding box [tl_col, tl_row, br_col, br_row] ??
# numNONSE of each people
# bb_size of each people
# bb_center of bounding box
# scale_provided
# 1/右肩，2/右肘，3/右腕，4/左肩，5/左肘，6/左腕，7/右髋，8/右膝，9/右踝，10/左髋，11/左膝，12/左踝，13/头顶，14/脖子
# joints
# MSCOCO   0: nonsense  1: not visible  2: visible
# AIC      3: nonsense  1: visible      2: not visible

min_size = 32 * 32
json_file = '../dataset/annotations/person_keypoints_valid2017.json'
img_flie = '/home/ghy/0a708bc04e70073ab9ae632e227b06dc25e2c59e.jpg'
img_id = '0a708bc04e70073ab9ae632e227b06dc25e2c59e'
img_list = [img_flie]
joints_num = 14
val_size = 2000
# load the keypoints annotations
with open(json_file, 'r') as json_object:
    ky_annotations = json.load(json_object)

print('ky_annotations include {} images.'.format(len(ky_annotations)))

joint_all = []  # target to return
for i in ky_annotations:
    for j in i['keypoint_annotations'].keys():
        pass
        """
        kp_anno0 = i['keypoint_annotations'][j]
        kp_type = kp_anno0[2::3]
        numNONSE = kp_type.count(3)
        print(kp_anno0)
        print(kp_type)
        print('There is(are) {} keypoint(s) not labelled.'.format(numNONSE))
        input('enter')
        """

    if i['image_id'] == img_id:
        print('Image has been found.')
        image = cv2.imread('../dataset/valid2017/'+i['image_id']+'.jpg')
        h, w, c = image.shape
        print('height: ', h, '; width: ', w, '; channels: ', c)
        numPeople = len(i['keypoint_annotations'])
        print('There are {} people in this image.'.format(numPeople))

        for j in i['keypoint_annotations'].keys():
            print(j)
            # find valid number of keypoints
            kp_anno = i['keypoint_annotations'][j]
            if len(kp_anno) != 42:
                print('    WARNING: A keypoint_annotation containing incomplete infos is found!')

            # print(kp_anno)
            kp_type = kp_anno[2::3]
            numNONSE = kp_type.count(3)
            print('    There is(are) {} keypoint(s) not labelled.'.format(numNONSE))

            # find the size of bb
            bb_anno = i['human_annotations'][j]
            print('    Bounding bos is ', bb_anno)
            bb_size_w = bb_anno[2] - bb_anno[0] + 1
            bb_size_h = bb_anno[3] - bb_anno[1] + 1
            bb_size = bb_size_h * bb_size_w
            bb_center = [(bb_anno[0] + bb_anno[2]) // 2,
                         (bb_anno[1] + bb_anno[3]) // 2]
            print('    (Bounding Box) Height: {}; Width: {}; Size: {}; Center: {}'
                  .format(bb_size_h, bb_size_w, bb_size, bb_center))

            joints = np.zeros((17, 3))
            for k in range(joints_num):
                joints[k, 0] = kp_anno[k*3]
                joints[k, 1] = kp_anno[k*3 + 1]

                if kp_anno[k * 3 + 2] == 1:    # visible
                    joints[k, 2] = 1
                elif kp_anno[k * 3 + 2] == 2:  # invisible
                    joints[k, 2] = 0
                else:                          # none
                    joints[k, 2] = 2

                # print(joints[k, :])

            scaled_provided = (bb_anno[3] - bb_anno[1] + 1) / 368

            # organize
        count = 0
        if numPeople > 0:  # this number could be smaller than the one in the annotation coz some annos may not be valid.
            joint_all.append(dict())
            joint_all[count]["dataset"] = 'AIC'

            if count < val_size:
                isValidation = 1
            else:
                isValidation = 0

            joint_all[count]["isValidation"] = isValidation

            joint_all[count]["img_width"] = w
            joint_all[count]["img_height"] = h
            joint_all[count]["image_id"] = img_id
            joint_all[count]["annolist_index"] = idx_img

            joint_all[count]["img_paths"] = os.path.join(images_dir, '%012d.jpg' % img_id)
            joint_all[count]["mask_miss_paths"] = os.path.join(masks_dir,
                                                               'mask_miss_%012d.png' % img_id)
            joint_all[count]["mask_all_paths"] = os.path.join(masks_dir,
                                                              'mask_all_%012d.png' % img_id)

            joint_all[count]["objpos"] = persons[0]["objpos"]
            joint_all[count]["bbox"] = persons[0]["bbox"]
            joint_all[count]["segment_area"] = persons[0]["segment_area"]
            joint_all[count]["num_keypoints"] = persons[0]["num_keypoints"]
            joint_all[count]["joint_self"] = persons[0]["joint"]
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



