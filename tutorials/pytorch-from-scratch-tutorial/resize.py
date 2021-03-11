"""
This script makes all of fold B's images the same size as fold A's.
"""
import argparse
"""
The make_dataset.py script creates a dataset that is almost usable,
but the dreamified images are not quite the right shape. We need them to
be identical in shape to the originals.

Run this script on the coco-dreamified directory to do this.

Then you can use:

python combine_A_and_B.py --fold_A coco-dreamified/A --fold_B coco-dreamified/B --fold_AB coco-dreamified-combined
"""
import cv2
import os
import PIL.Image
import numpy as np
from tqdm import tqdm

def resize_b_to_a(dpath_a, dpath_b):
    dpaths = [p for p in zip(os.listdir(dpath_a), os.listdir(dpath_b))]
    for name_a, name_b in tqdm(dpaths):
        path_a = os.path.join(dpath_a, name_a)
        img_a = np.array(PIL.Image.open(path_a))

        path_b = os.path.join(dpath_b, name_b)
        img_b = np.array(PIL.Image.open(path_b))

        if img_a.shape != img_b.shape:
            img_b = cv2.resize(img_b, dsize=img_a.shape[:-1][::-1], interpolation=cv2.INTER_CUBIC)
            img_b = PIL.Image.fromarray(np.array(img_b))
            img_b.save(path_b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dpath", type=str, help="Path to the dataset of images which need resizing.")
    args = parser.parse_args()

    train_a = os.path.join(args.dpath, "A", "train")
    val_a = os.path.join(args.dpath, "A", "val")
    train_b = os.path.join(args.dpath, "B", "train")
    val_b = os.path.join(args.dpath, "B", "val")

    print("Training split...")
    resize_b_to_a(train_a, train_b)

    print("Validation split...")
    resize_b_to_a(val_a, val_b)