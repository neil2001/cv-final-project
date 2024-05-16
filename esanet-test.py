# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from ESANet.src.args import ArgumentParserRGBDSegmentation
from ESANet.src.build_model import build_model
from ESANet.src.prepare_data import prepare_data

import sys


def _load_img(fp):
    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    # x, y, w, h = 64, 108, 512, 424 
    # img = img[y:y+h, x:x+w]
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help='Path to the checkpoint of the trained model.')
    parser.add_argument('--depth_scale', type=float,
                        default=1.0,
                        help='Additional depth scaling factor to apply.')
    args = parser.parse_args()

    # print(args)

    # sys.exit(0)

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    # get samples
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'samples')
    rgb_filepaths = sorted(glob(os.path.join(basepath, 'rgb.*')))
    depth_filepaths = sorted(glob(os.path.join(basepath, 'depth.*')))
    assert args.modality == 'rgbd', "Only RGBD inference supported so far"
    assert len(rgb_filepaths) == len(depth_filepaths)
    filepaths = zip(rgb_filepaths, depth_filepaths)

    # inference
    for fp_rgb, fp_depth in filepaths:
        # load sample
        img_rgb = _load_img(fp_rgb)
        img_depth = _load_img(fp_depth).astype('float32') * args.depth_scale
        h, w, _ = img_rgb.shape

        # img_depth = img_depth / 255.0
        # img_depth = np.mean(img_depth, axis=-1)

        print("img shape:", img_depth.shape) # img rgb shape: (424, 512, 3) depth (424, 512)

        # preprocess sample
        sample = preprocessor({'image': img_rgb, 'depth': img_depth})

        # add batch axis and copy to device
        image = sample['image'][None].to(device)
        depth = sample['depth'][None].to(device)

        # depth = depth[..., 0]

        # apply network
        pred = model(image, depth)
        pred = F.interpolate(pred, (h, w),
                             mode='bilinear', align_corners=False)
        pred = torch.argmax(pred, dim=1)
        pred = pred.cpu().numpy().squeeze().astype(np.uint8)

        # show result
        pred_colored = dataset.color_label(pred, with_void=False)
        fig, axs = plt.subplots(1, 3, figsize=(16, 3))
        [ax.set_axis_off() for ax in axs.ravel()]
        axs[0].imshow(img_rgb)
        axs[1].imshow(img_depth, cmap='gray')
        axs[2].imshow(pred_colored)

        plt.suptitle(f"Image: ({os.path.basename(fp_rgb)}, "
                     f"{os.path.basename(fp_depth)}), Model: {args.ckpt_path}")
        plt.savefig('./result.jpg', dpi=150)
        plt.show()
