import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer as vits


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    #parser.add_argument('--arch', default='vit_small', type=str,                                                       # 原代码
    #    choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--arch', default='vit_tiny', type=str,                                                         # 此处进行了修改
                        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    #parser.add_argument('--pretrained_weights', default='', type=str,                                                  # 原代码
    #    help="Path to pretrained weights to load.")
    parser.add_argument('--pretrained_weights', default='./train-t2-4for-0.997/checkpoint.pth', type=str,
                        help="Path to pretrained weights to load.")
    #parser.add_argument("--checkpoint_key", default="teacher", type=str,                                              # 原代码
    #    help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--checkpoint_key1", default="teacher1", type=str,                                              # 此处进行了修改
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--checkpoint_key2", default="teacher2", type=str,                                              # 此处进行了修改
                        help='Key to use in the checkpoint (example: "teacher")')
    #parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")                 
    #parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')                        # 原代码
    parser.add_argument('--output_dir1', default='attention-teacher1-t2', help='Path where to save visualizations.')                        # 此处进行了修改
    parser.add_argument('--output_dir2', default='attention-teacher2-t2', help='Path where to save visualizations.')                        # 此处进行了修改
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    #model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)                                        # 原代码
    model1 = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)                                        # 此处进行了修改
    model2 = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)                                        # 此处进行了修改
    #for p in model.parameters():                                                                                       # 原代码
    #    p.requires_grad = False
    for p in model1.parameters():                                                                                       # 此处进行了修改
        p.requires_grad = False
    for p in model2.parameters():                                                                                       # 此处进行了修改
        p.requires_grad = False
    #model.eval()                                                                                                       # 原代码
    #model.to(device)
    model1.eval()                                                                                                       # 此处进行了修改
    model1.to(device)
    model2.eval()                                                                                                       # 此处进行了修改
    model2.to(device)
    utils.load_pretrained_weights(model1, args.pretrained_weights, args.checkpoint_key1, args.arch,
                                  args.patch_size)                                                                      # 此处进行了修改
    utils.load_pretrained_weights(model2, args.pretrained_weights, args.checkpoint_key2, args.arch,
                                  args.patch_size)                                                                      # 此处进行了修改

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        #print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        #print("Since no image path have been provided, we take the first image in our paper.")
        #response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        response = requests.get("http://m.qpic.cn/psc?/V53mbERq3lU0kJ1E4sEy2ov0ue10cyie/ruAMsa53pVQWN7FLK88i5gVpcX9cZRjWO*ETSURSXng5ie2mU2N5doaVzrB*W9JwPsMeX.fz2hrdWjYiHCO5xpo88YpXDYwqTCZGVZcC5qY!/mnull&bo=dwH0AQAAAAABB6M!&rf=photolist&t=5")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    #attentions = model.get_last_selfattention(img.to(device))                                                          # 原代码
    attentions1 = model1.get_last_selfattention(img.to(device))                                                          # 此处进行了修改
    attentions2 = model2.get_last_selfattention(img.to(device))                                                          # 此处进行了修改

    #nh = attentions.shape[1] # number of head                                                                          # 原代码
    nh1 = attentions1.shape[1]  # number of head                                                                          # 此处进行了修改
    nh2 = attentions2.shape[1]  # number of head                                                                          # 此处进行了修改

    # we keep only the output patch attention
    #attentions = attentions[0, :, 0, 1:].reshape(nh, -1)                                                               # 原代码
    attentions1 = attentions1[0, :, 0, 1:].reshape(nh1, -1)                                                              # 此处进行了修改
    attentions2 = attentions2[0, :, 0, 1:].reshape(nh2, -1)                                                              # 此处进行了修改

    #if args.threshold is not None:                                                                                     # 原代码
        # we keep only a certain percentage of the mass
    #    val, idx = torch.sort(attentions)
    #    val /= torch.sum(val, dim=1, keepdim=True)
    #    cumval = torch.cumsum(val, dim=1)
    #    th_attn = cumval > (1 - args.threshold)
    #    idx2 = torch.argsort(idx)
    #    for head in range(nh):
    #        th_attn[head] = th_attn[head][idx2[head]]
    #    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
    #    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    if args.threshold is not None:                                                                                       # 此处进行了修改
        # we keep only a certain percentage of the mass
        val1, idx1 = torch.sort(attentions1)
        val1 /= torch.sum(val1, dim=1, keepdim=True)
        cumval1 = torch.cumsum(val1, dim=1)
        th_attn1 = cumval1 > (1 - args.threshold)
        idx21 = torch.argsort(idx1)
        for head in range(nh1):
            th_attn1[head] = th_attn1[head][idx21[head]]
        th_attn1 = th_attn1.reshape(nh1, w_featmap, h_featmap).float()
        # interpolate
        th_attn1 = nn.functional.interpolate(th_attn1.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    if args.threshold is not None:                                                                                      # 此处进行了修改
        # we keep only a certain percentage of the mass
        val2, idx2 = torch.sort(attentions2)
        val2 /= torch.sum(val2, dim=1, keepdim=True)
        cumval2 = torch.cumsum(val2, dim=1)
        th_attn2 = cumval2 > (1 - args.threshold)
        idx22 = torch.argsort(idx2)
        for head in range(nh2):
            th_attn2[head] = th_attn2[head][idx22[head]]
        th_attn2 = th_attn2.reshape(nh2, w_featmap, h_featmap).float()
        # interpolate
        th_attn2 = nn.functional.interpolate(th_attn2.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()


    #attentions = attentions.reshape(nh, w_featmap, h_featmap)                                                          # 原代码
    #attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
    attentions1 = attentions1.reshape(nh1, w_featmap, h_featmap)                                                          # 此处进行了修改
    attentions1 = nn.functional.interpolate(attentions1.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
        0].cpu().numpy()
    attentions2 = attentions2.reshape(nh2, w_featmap, h_featmap)                                                          # 此处进行了修改
    attentions2 = nn.functional.interpolate(attentions2.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
        0].cpu().numpy()

    # save attentions heatmaps
    #os.makedirs(args.output_dir, exist_ok=True)                                                                        # 原代码
    os.makedirs(args.output_dir1, exist_ok=True)                                                                        # 此处进行了修改
    os.makedirs(args.output_dir2, exist_ok=True)                                                                        # 此处进行了修改

    #torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join(args.output_dir1, "img.png"))                                             # 此处进行了修改
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True),
                                 os.path.join(args.output_dir2, "img.png"))                                             # 此处进行了修改

    #for j in range(nh):                                                                                                # 原代码
    #    fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
    #    plt.imsave(fname=fname, arr=attentions[j], format='png')
    #    print(f"{fname} saved.")
    for j in range(nh1):                                                                                                # 此处进行了修改
        fname = os.path.join(args.output_dir1, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions1[j], format='png')
        print(attentions1[j])
        print(f"{fname} saved.")
    for j in range(nh2):                                                                                                # 此处进行了修改
        fname = os.path.join(args.output_dir2, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions2[j], format='png')
        print(attentions2[j])
        print(f"{fname} saved.")

    #if args.threshold is not None:                                                                                     # 原代码
    #    image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
    #    for j in range(nh):
    #        display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
    if args.threshold is not None:                                                                                       # 此处进行了修改
        image = skimage.io.imread(os.path.join(args.output_dir1, "img.png"))
        for j in range(nh1):
            display_instances(image, th_attn1[j], fname=os.path.join(args.output_dir1, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)

    if args.threshold is not None:                                                                                       # 此处进行了修改
        image = skimage.io.imread(os.path.join(args.output_dir2, "img.png"))
        for j in range(nh2):
            display_instances(image, th_attn2[j], fname=os.path.join(args.output_dir2, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)