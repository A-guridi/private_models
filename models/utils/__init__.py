from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import os
import scipy.io


def infer_num_classes(dir):
    """
    Infers the number of segmentation classes in a dataset
    :param dir: directory of the dataset
    :return: int, number of clases
    """
    if not Path(dir).is_dir():
        raise ValueError("Directory not found")

    bincount = []
    for mask in Path(dir).iterdir():
        m = Image.open(mask)
        m = np.array(m)
        bincount.append(np.unique(m))

    return len(np.unique(bincount))


def transform_gt(dir):
    sets = ["train", "val", "test"]
    for s in sets:
        lb_dir = os.path.join(dir, "groundTruth", s)
        nwdir = os.path.join(dir, "label", s)
        Path(nwdir).mkdir(exist_ok=True, parents=True)
        for gt in Path(lb_dir).iterdir():
            m = scipy.io.loadmat(str(gt))["groundTruth"]
            print(np.array(m).shape)
            mask = Image.fromarray(np.array(m))
            f_name, fex = os.path.splitext(os.path.basename(str(gt)))
            mask.save(os.path.join(nwdir, f_name + ".png"))
    print("Ground truths transformed")


def create_datasets(dir):
    if not Path(dir).is_dir():
        raise ValueError("Directory not found")

    sets = ["train", "val", "test"]
    lb_sets = ["train_labels", "val_labels", "test_labels"]
    img_set = []
    lab_set = []

    for idx, (ts, ls) in enumerate(zip(sets, lb_sets)):
        im_dir = os.path.join(dir, ts)
        lab_dir = os.path.join(dir, ls)
        img_set[idx] = [im for im in Path(im_dir).iterdir()]
        lab_set[idx] = [lb for lb in Path(lab_dir).iterdir()]

    train_df = pd.DataFrame({"image": img_set[0],
                             "label": lab_set[0]})
    val_df = pd.DataFrame({"image": img_set[1],
                           "label": lab_set[1]})
    test_df = pd.DataFrame({"image": img_set[2],
                            "label": lab_set[2]})
    return train_df, val_df, test_df
