import os

import h5py
import matplotlib.pyplot as plt
import pandas as pd


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr[()][i].item()][()][0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr[()][0][0]]
        attrs[key] = values
    return attrs


def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file, 'r')
    print("All data length: " + str(f['/digitStruct/bbox'].shape[0]))
    all_rows = []
    print('image bounding box data construction starting...')
    bbox_df = pd.DataFrame([], columns=['height', 'img_name', 'label', 'left', 'top', 'width'])
    #     print("Original bbox_df:")
    #     print(bbox_df)
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        row_dict['img_name'] = img_name
        all_rows.append(row_dict)
        bbox_df = pd.concat([bbox_df, pd.DataFrame.from_dict(row_dict, orient='columns')])
    bbox_df['bottom'] = bbox_df['top'] + bbox_df['height']
    bbox_df['right'] = bbox_df['left'] + bbox_df['width']
    print('finished image bounding box data construction...')
    return bbox_df


if __name__ == "__main__":

    train_dir = "images/train"

    img_bbox_data = img_boundingbox_data_constructor(os.path.join(train_dir, 'digitStruct.mat'))
    img_bbox_data_grouped = img_bbox_data.groupby('img_name')

    print(len(img_bbox_data_grouped))
    for i in range(len(img_bbox_data_grouped)):
        f = open(os.path.join("labels/train", str(i + 1) + ".txt"), "w")
        img_name = str(i + 1) + ".png"
        img = plt.imread(os.path.join(train_dir, img_name))
        h = img.shape[0]
        w = img.shape[1]
        bbox = img_bbox_data_grouped.get_group(img_name)

        for i in range(len(bbox)):

            label = int(bbox['label'][i])
            x_center = (bbox['left'][i] + bbox['right'][i]) // 2
            y_center = (bbox['top'][i] + bbox['bottom'][i]) // 2
            width = bbox['width'][i]
            height = bbox['height'][i]

            f.write(str(label - 1) + " " + str(x_center / w) + " " + str(y_center / h) + " " +
                    str(width / w) + " " + str(height / h))
            if i < len(bbox) - 1:
                f.write("\n")
        f.close()

