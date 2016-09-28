#!/usr/bin/env python

"""create_patches.py

Create labeled characters for training the character classification network.

Author: Amit Aides, Ahmad Kiswani
License: See attached license file
"""

from __future__ import division
import AUVSItargets
import AUVSItargets.global_settings as gs
import random
import shutil
from joblib import Parallel, delayed
import cv2
import glob
import os
import argparse

RESIZED = False
if RESIZED:
    K = gs.resized_K
    SUB_PATH = 'resized_images'
else:
    K = gs.K
    SUB_PATH = 'renamed_images'


def main(jobs):
    #
    # Setup the paths.
    #
    imgs_paths = sorted(
            glob.glob(os.path.join(gs.DATA_PATH, SUB_PATH, '*.jpg'))
        )
    img_names = [
            os.path.splitext(os.path.split(path)[1])[0] for path in imgs_paths
        ]
    data_paths = [os.path.join(gs.DATA_PATH,
                                'flight_data',
                                'resized_'+name+'.json'
                                ) for name in img_names]

    #
    # Delete any old dst folder
    #
    dst_folder = os.path.join(gs.DATA_PATH, 'train_letter')
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)

    os.makedirs(dst_folder)

    img_index = 0
    for img_path, data_path in zip(imgs_paths, data_paths):

        print 'Extracting patches from image', img_path

        img = AUVSItargets.Image(img_path, data_path, K=K)
        patches = img.createPatches(patch_size=gs.PATCH_SIZE, patch_shift=1000)

        results = Parallel(n_jobs=jobs)(
            delayed(create_patch)(patch,
                                 img,
                                 img.latitude,
                                 img.longitude,
                                 img.yaw) for patch in patches
        )

        for mask, letter_label in results:
            if mask is None:
                continue

            filename = '{:07}'.format(img_index)
            img_index += 1

            cv2.imwrite(os.path.join(dst_folder, filename+'.jpg'), mask)
            with open(os.path.join(dst_folder, filename+'.txt'), 'w') as fp:
                fp.write('{}.jpg\t{}'.format(filename, letter_label))

def create_patch(patch, img, latitude, longitude, yaw):

    #
    # Letters are pasted on a rectangle to improve the segmentation accuracy.
    #
    letter_label = random.choice(gs.LETTER_LABELS)
    target_label = gs.SHAPE_LABELS.index('Rectangle')

    if letter_label != 'no target':
        if letter_label == 'rotated letter':
            dangle = random.randint(45, 315)
        else:
            dangle = random.randint(-5, 5)

        #
        # Paste a rotatedrandom target on the patch
        #
        target, _, _ = AUVSItargets.randomTarget(
            altitude=0,
            longitude=longitude,
            latitude=latitude,
            target_label=target_label,
            orientation=yaw+dangle
        )

        br = img.pastePatch(patch=patch, target=target)
        br = AUVSItargets.squareCoords(br, noise=False)
        patch = patch[br[1]:br[3], br[0]:br[2], ...]

    #
    # Mask out the letter and tight crop.
    #
    try:
        kmean_mask, _ = AUVSItargets.KMEANS.getLetterMask(patch)
        if kmean_mask is None:
            #
            # The segmentation failed.
            #
            return None, None

        mask = AUVSItargets.tightCrop(kmean_mask)
    except:
        return None, None

    print gs.LETTER_LABELS.index(letter_label)

    return mask, gs.LETTER_LABELS.index(letter_label)


if __name__ == '__main__':

    cmdline = argparse.ArgumentParser(usage="usage: ./{}"
                                            .format(os.path.basename(__file__)),
                                      description="Create letter patches")
    cmdline.add_argument("--jobs",
                         "-j",
                         action="store",
                         help="Number of cores to use (default=1).",
                         type=int,
                         dest="jobs",
                         default=1)

    args = cmdline.parse_args()

    main(jobs=args.jobs)
