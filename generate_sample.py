import json
import os
import numpy as np
from shutil import copyfile


def gen_sample_dataset(image_path, anno_path, num):
    json_path = anno_path
    img_path = image_path
    sample_nb = num

    # load the json file
    with open(json_path, 'r') as json_object:
        kp_json = json.load(json_object)

    # generate the original img file list
    img_files = os.listdir(img_path)
    img_files_cnt = len(img_files)
    sample_nb = min(sample_nb, img_files_cnt)

    # generate random sample file names
    shuffled_list = np.random.permutation(img_files_cnt)[:sample_nb]
    cp_list = [img_files[idx] for idx in shuffled_list]

    # make dir
    if img_path[-1] == '/':
        img_path_sample = img_path[:-1] + '_sample'
    else:
        img_path_sample = img_path[:-1] + '_sample'
    os.mkdir(img_path_sample)

    # copy img files
    for f in cp_list:
        copyfile(img_path+'/'+f, img_path_sample+'/'+f)

    # generate sample annotations
    cp_id_list = [i.split('.')[0] for i in cp_list]
    kp_list_sample = []
    for j in kp_json:
        if j['image_id'] in cp_id_list:
            kp_list_sample.append(j)

        if len(kp_list_sample) == len(cp_id_list):
            break

    # generate json file for sample annotations
    sample_json = img_path_sample+'_kp.json'

    with open(sample_json, 'w') as json_object:
        json.dump(kp_list_sample, json_object)


def check_sample(img_path, json_file):
    with open(json_file, 'r') as json_object:
        kp_json = json.load(json_object)

    kp_len = len(kp_json)
    print('The json file contains keypoint infos for {} images.'.format(kp_len))

    img_files = os.listdir(img_path)
    img_files = [i for i in img_files if i.split('.')[-1] == 'jpg']
    file_cnt = len(img_files)
    print('The image path contains {} images.'.format(file_cnt))
    if kp_len == file_cnt:
        print('The numbers match.')
    else:
        raise Exception('The numbers do not match !!')

    print('Now check if the ids match...')

    img_ids = [i.split('.')[0] for i in img_files]
    img_ids_pop = []
    for i in kp_json:
        try:
            idx = img_ids.index(i['image_id'])
        except ValueError:
            if i['image_id'] in img_ids_pop:
                raise Exception('image_id: %s repeated in json file.' %i['image_id000'])
            else:
                raise Exception('Check failed. Coz one image of which keypoints '
                                'are included in json file does not exit in image path.')

        img_ids_pop.append(img_ids.pop(idx))

    print('check passed.')
    return 0




