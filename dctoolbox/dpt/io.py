"""Utils for monoDepth.
"""
from __future__ import annotations

import cv2
import numpy as np
import re
import sys
import torch
from PIL import Image

adepallete = [
    0,
    0,
    0,
    120,
    120,
    120,
    180,
    120,
    120,
    6,
    230,
    230,
    80,
    50,
    50,
    4,
    200,
    3,
    120,
    120,
    80,
    140,
    140,
    140,
    204,
    5,
    255,
    230,
    230,
    230,
    4,
    250,
    7,
    224,
    5,
    255,
    235,
    255,
    7,
    150,
    5,
    61,
    120,
    120,
    70,
    8,
    255,
    51,
    255,
    6,
    82,
    143,
    255,
    140,
    204,
    255,
    4,
    255,
    51,
    7,
    204,
    70,
    3,
    0,
    102,
    200,
    61,
    230,
    250,
    255,
    6,
    51,
    11,
    102,
    255,
    255,
    7,
    71,
    255,
    9,
    224,
    9,
    7,
    230,
    220,
    220,
    220,
    255,
    9,
    92,
    112,
    9,
    255,
    8,
    255,
    214,
    7,
    255,
    224,
    255,
    184,
    6,
    10,
    255,
    71,
    255,
    41,
    10,
    7,
    255,
    255,
    224,
    255,
    8,
    102,
    8,
    255,
    255,
    61,
    6,
    255,
    194,
    7,
    255,
    122,
    8,
    0,
    255,
    20,
    255,
    8,
    41,
    255,
    5,
    153,
    6,
    51,
    255,
    235,
    12,
    255,
    160,
    150,
    20,
    0,
    163,
    255,
    140,
    140,
    140,
    250,
    10,
    15,
    20,
    255,
    0,
    31,
    255,
    0,
    255,
    31,
    0,
    255,
    224,
    0,
    153,
    255,
    0,
    0,
    0,
    255,
    255,
    71,
    0,
    0,
    235,
    255,
    0,
    173,
    255,
    31,
    0,
    255,
    11,
    200,
    200,
    255,
    82,
    0,
    0,
    255,
    245,
    0,
    61,
    255,
    0,
    255,
    112,
    0,
    255,
    133,
    255,
    0,
    0,
    255,
    163,
    0,
    255,
    102,
    0,
    194,
    255,
    0,
    0,
    143,
    255,
    51,
    255,
    0,
    0,
    82,
    255,
    0,
    255,
    41,
    0,
    255,
    173,
    10,
    0,
    255,
    173,
    255,
    0,
    0,
    255,
    153,
    255,
    92,
    0,
    255,
    0,
    255,
    255,
    0,
    245,
    255,
    0,
    102,
    255,
    173,
    0,
    255,
    0,
    20,
    255,
    184,
    184,
    0,
    31,
    255,
    0,
    255,
    61,
    0,
    71,
    255,
    255,
    0,
    204,
    0,
    255,
    194,
    0,
    255,
    82,
    0,
    10,
    255,
    0,
    112,
    255,
    51,
    0,
    255,
    0,
    194,
    255,
    0,
    122,
    255,
    0,
    255,
    163,
    255,
    153,
    0,
    0,
    255,
    10,
    255,
    112,
    0,
    143,
    255,
    0,
    82,
    0,
    255,
    163,
    255,
    0,
    255,
    235,
    0,
    8,
    184,
    170,
    133,
    0,
    255,
    0,
    255,
    92,
    184,
    0,
    255,
    255,
    0,
    31,
    0,
    184,
    255,
    0,
    214,
    255,
    255,
    0,
    112,
    92,
    255,
    0,
    0,
    224,
    255,
    112,
    224,
    255,
    70,
    184,
    160,
    163,
    0,
    255,
    153,
    0,
    255,
    71,
    255,
    0,
    255,
    0,
    163,
    255,
    204,
    0,
    255,
    0,
    143,
    0,
    255,
    235,
    133,
    255,
    0,
    255,
    0,
    235,
    245,
    0,
    255,
    255,
    0,
    122,
    255,
    245,
    0,
    10,
    190,
    212,
    214,
    255,
    0,
    0,
    204,
    255,
    20,
    0,
    255,
    255,
    255,
    0,
    0,
    153,
    255,
    0,
    41,
    255,
    0,
    255,
    204,
    41,
    0,
    255,
    41,
    255,
    0,
    173,
    0,
    255,
    0,
    245,
    255,
    71,
    0,
    255,
    122,
    0,
    255,
    0,
    255,
    184,
    0,
    92,
    255,
    184,
    255,
    0,
    0,
    133,
    255,
    255,
    214,
    0,
    25,
    194,
    194,
    102,
    255,
    0,
    92,
    0,
    255,
]

citypallete = [
    128,
    64,
    128,
    244,
    35,
    232,
    70,
    70,
    70,
    102,
    102,
    156,
    190,
    153,
    153,
    153,
    153,
    153,
    250,
    170,
    30,
    220,
    220,
    0,
    107,
    142,
    35,
    152,
    251,
    152,
    70,
    130,
    180,
    220,
    20,
    60,
    255,
    0,
    0,
    0,
    0,
    142,
    0,
    0,
    70,
    0,
    60,
    100,
    0,
    80,
    100,
    0,
    0,
    230,
    119,
    11,
    32,
    128,
    192,
    0,
    0,
    64,
    128,
    128,
    64,
    128,
    0,
    192,
    128,
    128,
    192,
    128,
    64,
    64,
    0,
    192,
    64,
    0,
    64,
    192,
    0,
    192,
    192,
    0,
    64,
    64,
    128,
    192,
    64,
    128,
    64,
    192,
    128,
    192,
    192,
    128,
    0,
    0,
    64,
    128,
    0,
    64,
    0,
    128,
    64,
    128,
    128,
    64,
    0,
    0,
    192,
    128,
    0,
    192,
    0,
    128,
    192,
    128,
    128,
    192,
    64,
    0,
    64,
    192,
    0,
    64,
    64,
    128,
    64,
    192,
    128,
    64,
    64,
    0,
    192,
    192,
    0,
    192,
    64,
    128,
    192,
    192,
    128,
    192,
    0,
    64,
    64,
    128,
    64,
    64,
    0,
    192,
    64,
    128,
    192,
    64,
    0,
    64,
    192,
    128,
    64,
    192,
    0,
    192,
    192,
    128,
    192,
    192,
    64,
    64,
    64,
    192,
    64,
    64,
    64,
    192,
    64,
    192,
    192,
    64,
    64,
    64,
    192,
    192,
    64,
    192,
    64,
    192,
    192,
    192,
    192,
    192,
    32,
    0,
    0,
    160,
    0,
    0,
    32,
    128,
    0,
    160,
    128,
    0,
    32,
    0,
    128,
    160,
    0,
    128,
    32,
    128,
    128,
    160,
    128,
    128,
    96,
    0,
    0,
    224,
    0,
    0,
    96,
    128,
    0,
    224,
    128,
    0,
    96,
    0,
    128,
    224,
    0,
    128,
    96,
    128,
    128,
    224,
    128,
    128,
    32,
    64,
    0,
    160,
    64,
    0,
    32,
    192,
    0,
    160,
    192,
    0,
    32,
    64,
    128,
    160,
    64,
    128,
    32,
    192,
    128,
    160,
    192,
    128,
    96,
    64,
    0,
    224,
    64,
    0,
    96,
    192,
    0,
    224,
    192,
    0,
    96,
    64,
    128,
    224,
    64,
    128,
    96,
    192,
    128,
    224,
    192,
    128,
    32,
    0,
    64,
    160,
    0,
    64,
    32,
    128,
    64,
    160,
    128,
    64,
    32,
    0,
    192,
    160,
    0,
    192,
    32,
    128,
    192,
    160,
    128,
    192,
    96,
    0,
    64,
    224,
    0,
    64,
    96,
    128,
    64,
    224,
    128,
    64,
    96,
    0,
    192,
    224,
    0,
    192,
    96,
    128,
    192,
    224,
    128,
    192,
    32,
    64,
    64,
    160,
    64,
    64,
    32,
    192,
    64,
    160,
    192,
    64,
    32,
    64,
    192,
    160,
    64,
    192,
    32,
    192,
    192,
    160,
    192,
    192,
    96,
    64,
    64,
    224,
    64,
    64,
    96,
    192,
    64,
    224,
    192,
    64,
    96,
    64,
    192,
    224,
    64,
    192,
    96,
    192,
    192,
    224,
    192,
    192,
    0,
    32,
    0,
    128,
    32,
    0,
    0,
    160,
    0,
    128,
    160,
    0,
    0,
    32,
    128,
    128,
    32,
    128,
    0,
    160,
    128,
    128,
    160,
    128,
    64,
    32,
    0,
    192,
    32,
    0,
    64,
    160,
    0,
    192,
    160,
    0,
    64,
    32,
    128,
    192,
    32,
    128,
    64,
    160,
    128,
    192,
    160,
    128,
    0,
    96,
    0,
    128,
    96,
    0,
    0,
    224,
    0,
    128,
    224,
    0,
    0,
    96,
    128,
    128,
    96,
    128,
    0,
    224,
    128,
    128,
    224,
    128,
    64,
    96,
    0,
    192,
    96,
    0,
    64,
    224,
    0,
    192,
    224,
    0,
    64,
    96,
    128,
    192,
    96,
    128,
    64,
    224,
    128,
    192,
    224,
    128,
    0,
    32,
    64,
    128,
    32,
    64,
    0,
    160,
    64,
    128,
    160,
    64,
    0,
    32,
    192,
    128,
    32,
    192,
    0,
    160,
    192,
    128,
    160,
    192,
    64,
    32,
    64,
    192,
    32,
    64,
    64,
    160,
    64,
    192,
    160,
    64,
    64,
    32,
    192,
    192,
    32,
    192,
    64,
    160,
    192,
    192,
    160,
    192,
    0,
    96,
    64,
    128,
    96,
    64,
    0,
    224,
    64,
    128,
    224,
    64,
    0,
    96,
    192,
    128,
    96,
    192,
    0,
    224,
    192,
    128,
    224,
    192,
    64,
    96,
    64,
    192,
    96,
    64,
    64,
    224,
    64,
    192,
    224,
    64,
    64,
    96,
    192,
    192,
    96,
    192,
    64,
    224,
    192,
    192,
    224,
    192,
    32,
    32,
    0,
    160,
    32,
    0,
    32,
    160,
    0,
    160,
    160,
    0,
    32,
    32,
    128,
    160,
    32,
    128,
    32,
    160,
    128,
    160,
    160,
    128,
    96,
    32,
    0,
    224,
    32,
    0,
    96,
    160,
    0,
    224,
    160,
    0,
    96,
    32,
    128,
    224,
    32,
    128,
    96,
    160,
    128,
    224,
    160,
    128,
    32,
    96,
    0,
    160,
    96,
    0,
    32,
    224,
    0,
    160,
    224,
    0,
    32,
    96,
    128,
    160,
    96,
    128,
    32,
    224,
    128,
    160,
    224,
    128,
    96,
    96,
    0,
    224,
    96,
    0,
    96,
    224,
    0,
    224,
    224,
    0,
    96,
    96,
    128,
    224,
    96,
    128,
    96,
    224,
    128,
    224,
    224,
    128,
    32,
    32,
    64,
    160,
    32,
    64,
    32,
    160,
    64,
    160,
    160,
    64,
    32,
    32,
    192,
    160,
    32,
    192,
    32,
    160,
    192,
    160,
    160,
    192,
    96,
    32,
    64,
    224,
    32,
    64,
    96,
    160,
    64,
    224,
    160,
    64,
    96,
    32,
    192,
    224,
    32,
    192,
    96,
    160,
    192,
    224,
    160,
    192,
    32,
    96,
    64,
    160,
    96,
    64,
    32,
    224,
    64,
    160,
    224,
    64,
    32,
    96,
    192,
    160,
    96,
    192,
    32,
    224,
    192,
    160,
    224,
    192,
    96,
    96,
    64,
    224,
    96,
    64,
    96,
    224,
    64,
    224,
    224,
    64,
    96,
    96,
    192,
    224,
    96,
    192,
    96,
    224,
    192,
    0,
    0,
    0,
]


def _get_voc_pallete(num_cls):
    n = num_cls
    pallete = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    return pallete


vocpallete = _get_voc_pallete(256)


def get_mask_pallete(npimg, dataset="detail"):
    """Get image color pallete for visualizing masks"""
    # recovery boundary
    if dataset == "pascal_voc":
        npimg[npimg == 21] = 255
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    if dataset == "ade20k":
        out_img.putpalette(adepallete)
    elif dataset == "citys":
        out_img.putpalette(citypallete)
    elif dataset in ("detail", "pascal_voc", "pascal_aug"):
        out_img.putpalette(vocpallete)
    return out_img


def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale


def write_pfm(path, image, scale=1):
    """Write pfm file.

    Args:
        path (str): pathto file
        image (array): data
        scale (int, optional): Scale. Defaults to 1.
    """

    with open(path, "wb") as file:
        color = None

        if image.dtype.name != "float32":
            raise Exception("Image dtype must be float32.")

        image = np.flipud(image)

        if len(image.shape) == 3 and image.shape[2] == 3:  # color image
            color = True
        elif (
            len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
        ):  # greyscale
            color = False
        else:
            raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

        file.write("PF\n" if color else "Pf\n".encode())
        file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

        endian = image.dtype.byteorder

        if endian == "<" or endian == "=" and sys.byteorder == "little":
            scale = -scale

        file.write("%f\n".encode() % scale)

        image.tofile(file)


def read_image(path) -> np.ndarray:
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

    return img


def resize_image(img):
    """Resize image and make it fit for network.

    Args:
        img (array): image

    Returns:
        tensor: data ready for network
    """
    height_orig = img.shape[0]
    width_orig = img.shape[1]

    if width_orig > height_orig:
        scale = width_orig / 384
    else:
        scale = height_orig / 384

    height = (np.ceil(height_orig / scale / 32) * 32).astype(int)
    width = (np.ceil(width_orig / scale / 32) * 32).astype(int)

    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    img_resized = (
        torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).contiguous().float()
    )
    img_resized = img_resized.unsqueeze(0)

    return img_resized


def resize_depth(depth, width, height):
    """Resize depth map and bring to CPU (numpy).

    Args:
        depth (tensor): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = torch.squeeze(depth[0, :, :, :]).to("cpu")

    depth_resized = cv2.resize(
        depth.numpy(), (width, height), interpolation=cv2.INTER_CUBIC
    )

    return depth_resized


def write_depth(path, depth, bits=1, absolute_depth=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    write_pfm(path + ".pfm", depth.astype(np.float32))

    if absolute_depth:
        out = depth
    else:
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2 ** (8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = np.zeros(depth.shape, dtype=depth.dtype)

    if bits == 1:
        cv2.imwrite(
            path + ".png", out.astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )
    elif bits == 2:
        cv2.imwrite(
            path + ".png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )

    return


def write_segm_img(path, image, labels, palette="detail", alpha=0.5):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        image (array): input image
        labels (array): labeling of the image
    """

    mask = get_mask_pallete(labels, "ade20k")

    img = Image.fromarray(np.uint8(255 * image)).convert("RGBA")
    seg = mask.convert("RGBA")

    out = Image.blend(img, seg, alpha)

    out.save(path + ".png")

    return
