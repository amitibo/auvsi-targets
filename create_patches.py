#!/usr/bin/env python

"""create_patches.py

Create labeled samples for training the shape classification network.

Author: Amit Aides, Ahmad Kiswani
License: See attached license file
"""

from __future__ import division
import AUVSItargets
import AUVSItargets.global_settings as gs
import shutil
import cv2
import glob
import os
import argparse


def main(visualize):
    #
    # Create paths
    #
    imgs_paths = sorted(glob.glob(
        os.path.join(gs.DATA_PATH, 'resized_images', '*.jpg')))
    img_names = [
            os.path.splitext(os.path.split(path)[1])[0] for path in imgs_paths
        ]
    data_paths = [
            os.path.join(gs.DATA_PATH,
                         'flight_data',
                         name+'.json') for name in img_names
        ]
    dst_folder = os.path.join(gs.DATA_PATH, 'train_images')

    #
    # Prepare empty destination folder (where training data will be stored).
    #
    if os.path.exists(dst_folder):
        shutil.rmtree(dst_folder)

    os.makedirs(dst_folder)

    #
    # Load image and image data
    #
    with_target_flag = 0
    img_index = 0
    for shape_img_path, shape_data_path, empty_img_path, empty_data_path in zip(
        imgs_paths,
        data_paths,
        imgs_paths,
        data_paths):

        print 'Extracting patches from image', shape_img_path

        shape_img = AUVSItargets.Image(shape_img_path,
            shape_data_path,
            K=gs.resized_K)
        empty_img = AUVSItargets.Image(empty_img_path,
            empty_data_path,
            K=gs.resized_K)

        shape_patches = shape_img.createPatches(patch_size=gs.PATCH_SIZE,
            patch_shift=20)
        empty_patches = empty_img.createRandomSizedPatches(
            patch_size_range=gs.RANDOM_PATCH_SIZE_RANGE, patch_shift=20)

        for patch_pair in zip(empty_patches, shape_patches):
            patch = patch_pair[with_target_flag]
            if with_target_flag:
                #
                # Paste a random target on the patch
                #
                target, target_label, _ = AUVSItargets.randomTarget(
                    altitude=0,
                    longitude=shape_img.longitude,
                    latitude=shape_img.latitude
                )
                coords = shape_img.pastePatch(patch=patch, target=target)
                coords = AUVSItargets.squareCoords(coords, noise=True)
                original_patch = patch.copy()
                patch = patch[coords[1]:coords[3], coords[0]:coords[2], ...]
            else:
                original_patch = patch.copy()

            patch = cv2.resize(patch, dsize=gs.CLASSIFIER_PATCH_SIZE)

            if visualize:
                cv2.namedWindow('original patch', flags=cv2.WINDOW_NORMAL)
                cv2.imshow('original patch', original_patch)

                cv2.namedWindow('patch', flags=cv2.WINDOW_NORMAL)
                cv2.imshow('patch', patch)
                cv2.waitKey(0)

            if with_target_flag:
                label = target_label
            else:
                label = len(gs.SHAPE_LABELS)-1
            with_target_flag = 1 - with_target_flag

            filename = '{:07}'.format(img_index)
            img_index += 1

            cv2.imwrite(os.path.join(dst_folder, filename+'.jpg'), patch)
            with open(os.path.join(dst_folder, filename+'.txt'), 'w') as fp:
                fp.write('{}.jpg\t{}'.format(filename, label))

    if visualize:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cmdline = argparse.ArgumentParser(usage="usage: ./{}"
                                            .format(os.path.basename(__file__)),
                                      description="Create target patches")

    cmdline.add_argument("--visualize",
                         action="store_true",
                         help="Visualize outputs.",
                         dest="visualize",
                         default=False)

    args = cmdline.parse_args()

    main(visualize=args.visualize)
