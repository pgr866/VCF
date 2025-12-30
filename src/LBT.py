'''Exploiting spatial redundancy with the Lapped Biorthogonal Transform (LBT).'''

import numpy as np
import logging
import struct
import importlib
import cv2
import os

# Import resources
from DCT2D.block_DCT import analyze_image as space_analyze
from DCT2D.block_DCT import synthesize_image as space_synthesize
from DCT2D.block_DCT import get_subbands, get_blocks
from color_transforms.YCoCg import from_RGB, to_RGB
from information_theory import distortion
import main

# Write description before importing parser (parser.py tries to read it)
os.makedirs("/tmp", exist_ok=True)
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)

# Load local parser safely
import importlib.util
import sys
parser_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parser.py")
spec = importlib.util.spec_from_file_location("local_parser", parser_path)
local_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_parser)

# Inject local parser into sys.modules for other modules
sys.modules['parser'] = local_parser

# Default settings
default_block_size = 8
default_CT = "YCoCg"
perceptual_quantization = False
disable_subbands = False

# Add parser arguments
local_parser.parser_encode.add_argument("-B", "--block_size_LBT", type=local_parser.int_or_str, 
                                        help=f"Block size (default: {default_block_size})", 
                                        default=default_block_size)
local_parser.parser_encode.add_argument("-t", "--color_transform", type=local_parser.int_or_str, 
                                        help=f"Color transform (default: \"{default_CT}\")", 
                                        default=default_CT)
local_parser.parser_encode.add_argument("-p", "--perceptual_quantization", action='store_true', 
                                        help=f"Use perceptual quantization (default: \"{perceptual_quantization}\")", 
                                        default=perceptual_quantization)
local_parser.parser_encode.add_argument("-x", "--disable_subbands", action='store_true', 
                                        help=f"Disable the coefficients reordering in subbands (default: \"{disable_subbands}\")", 
                                        default=disable_subbands)

local_parser.parser_decode.add_argument("-B", "--block_size_LBT", type=local_parser.int_or_str, 
                                        help=f"Block size (default: {default_block_size})", 
                                        default=default_block_size)
local_parser.parser_decode.add_argument("-t", "--color_transform", type=local_parser.int_or_str, 
                                        help=f"Color transform (default: \"{default_CT}\")", 
                                        default=default_CT)
local_parser.parser_decode.add_argument("-p", "--perceptual_quantization", action='store_true', 
                                        help=f"Use perceptual dequantization (default: \"{perceptual_quantization}\")", 
                                        default=perceptual_quantization)
local_parser.parser_decode.add_argument("-x", "--disable_subbands", action='store_true', 
                                        help=f"Disable the coefficients reordering in subbands (default: \"{disable_subbands}\")", 
                                        default=disable_subbands)

args = local_parser.parser.parse_known_args()[0]
CT = importlib.import_module(args.color_transform)

# --- LBT Kernels ---

def lbt_kernel_1d(data, inverse=False):
    """
    1D LBT pre/post filter kernel.

    Parameters:
        data (np.ndarray): 1D signal block.
        inverse (bool): Apply inverse LBT if True.

    Returns:
        np.ndarray: Transformed 1D signal block.
    """
    N = len(data)
    half = N // 2
    out = data.astype(np.float64).copy()
    alpha = 0.15 * np.pi
    c, s = np.cos(alpha), np.sin(alpha)

    if not inverse:
        for i in range(half):
            a, b = out[i], out[N - 1 - i]
            out[i] = (a + b) * 0.5
            out[N - 1 - i] = (b - a)
            if i == 0:
                x, y = out[i], out[N-1-i]
                out[i] = c*x + s*y
                out[N-1-i] = -s*x + c*y
    else:
        for i in range(half):
            if i == 0:
                x, y = out[i], out[N-1-i]
                out[i] = c*x - s*y
                out[N-1-i] = s*x + c*y
            a, b = out[i], out[N - 1 - i]
            out[i] = a - b * 0.5
            out[N - 1 - i] = a + b * 0.5
    return out

def apply_lbt_2d(img, block_size, inverse=False):
    """
    Apply 2D LBT transform on a 3D image.

    Parameters:
        img (np.ndarray): Input image (height x width x channels).
        block_size (int): Block size for LBT.
        inverse (bool): Apply inverse LBT if True.

    Returns:
        np.ndarray: LBT-transformed image.
    """
    h, w, chans = img.shape
    out = img.astype(np.float64).copy()
    N = block_size
    offset = N // 2

    for c in range(chans):
        for i in range(h):
            for j in range(offset, w - offset, N):
                if j + N <= w:
                    out[i, j:j+N, c] = lbt_kernel_1d(out[i, j:j+N, c], inverse)
        for j in range(w):
            for i in range(offset, h - offset, N):
                if i + N <= h:
                    out[i:i+N, j, c] = lbt_kernel_1d(out[i:i+N, j, c], inverse)
    return out.astype(np.float32)

# --- LBT CoDec Class ---

class CoDec(CT.CoDec):
    """
    LBT + DCT codec class.
    """

    def __init__(self, args):
        logging.debug("trace")
        super().__init__(args)
        self.block_size = args.block_size_LBT
        self.use_lbt = True
        logging.debug(f"block_size = {self.block_size}")

        if args.perceptual_quantization:
            self.Y_QSSs = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
                                    [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
                                    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
                                    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]).astype(np.uint8)
            self.C_QSSs = np.array([[17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],
                                    [24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],
                                    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
                                    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99]]).astype(np.uint8)
            inter = cv2.INTER_AREA if self.block_size < 8 else cv2.INTER_LINEAR
            self.C_QSSs = cv2.resize(self.C_QSSs, (self.block_size, self.block_size), interpolation=inter)
            self.Y_QSSs = cv2.resize(self.Y_QSSs, (self.block_size, self.block_size), interpolation=inter)

        self.offset = 128 if args.quantizer == "deadzone" else 0

    def pad_and_center_to_multiple_of_block_size(self, img):
        """
        Pads a 3D image to multiples of the block size, centering the input.

        Parameters:
            img (np.ndarray): Input image (H x W x C).

        Returns:
            np.ndarray: Padded image.
        """
        logging.debug("trace")
        if img.ndim != 3:
            raise ValueError("Input image must be a 3D array.")
        self.original_shape = img.shape
        h, w, c = img.shape
        th = (h + self.block_size - 1) // self.block_size * self.block_size
        tw = (w + self.block_size - 1) // self.block_size * self.block_size
        ph, pw = th - h, tw - w
        top, left = ph // 2, pw // 2
        padded_img = np.pad(img, ((top, ph-top), (left, pw-left), (0,0)), mode='constant', constant_values=0)
        return padded_img

    def remove_padding(self, img):
        """
        Remove padding from a padded 3D image.

        Parameters:
            img (np.ndarray): Padded image.

        Returns:
            np.ndarray: Original image.
        """
        logging.debug("trace")
        if img.ndim != 3:
            raise ValueError("Input must be 3D.")
        if self.original_shape is None:
            raise ValueError("Original shape not set.")
        oh, ow, _ = self.original_shape
        ph, pw, _ = img.shape
        top, left = (ph - oh)//2, (pw - ow)//2
        return img[top:top+oh, left:left+ow, :]

    def encode_fn(self, in_fn, out_fn):
        """
        Encode an image using LBT + DCT.

        Parameters:
            in_fn (str): Input image path.
            out_fn (str): Output code-stream path.

        Returns:
            int: Output size in bytes.
        """
        logging.debug("trace")
        img = self.encode_read_fn(in_fn).astype(np.float32)
        self.original_shape = img.shape
        img = self.pad_and_center_to_multiple_of_block_size(img)
        with open(f"{out_fn}_shape.bin", "wb") as f:
            f.write(struct.pack("iii", *self.original_shape))
        img -= self.offset

        ct_img = from_RGB(img)
        if self.use_lbt:
            ct_img = apply_lbt_2d(ct_img, self.block_size, inverse=False)
        dct_img = space_analyze(ct_img, self.block_size, self.block_size)

        if args.perceptual_quantization:
            blocks_in_y = int(img.shape[0]/self.block_size)
            blocks_in_x = int(img.shape[1]/self.block_size)
            for by in range(blocks_in_y):
                for bx in range(blocks_in_x):
                    block = dct_img[by*self.block_size:(by+1)*self.block_size,
                                    bx*self.block_size:(bx+1)*self.block_size, :]
                    block[..., 0] *= (self.Y_QSSs/121)
                    block[..., 1] *= (self.C_QSSs/99)
                    block[..., 2] *= (self.C_QSSs/99)
                    dct_img[by*self.block_size:(by+1)*self.block_size,
                            bx*self.block_size:(bx+1)*self.block_size, :] = block

        decom_img = dct_img if args.disable_subbands else get_subbands(dct_img, self.block_size, self.block_size)
        decom_k = self.quantize_decom(decom_img)
        decom_k += self.offset
        decom_k = self.compress(decom_k.astype(np.uint8))
        output_size = self.encode_write_fn(decom_k, out_fn)
        return output_size

    def encode(self, in_fn="/tmp/original.png", out_fn="/tmp/encoded"):
        return self.encode_fn(in_fn, out_fn)

    def decode_fn(self, in_fn, out_fn):
        """
        Decode a code-stream using LBT + DCT.

        Parameters:
            in_fn (str): Input code-stream path.
            out_fn (str): Output image path.

        Returns:
            int: Output image size in bytes.
        """
        logging.debug("trace")
        decom_k = self.decode_read_fn(in_fn)
        with open(f"{in_fn}_shape.bin", "rb") as f:
            self.original_shape = struct.unpack("iii", f.read(12))
        decom_k = self.decompress(decom_k).astype(np.int16)
        decom_k -= self.offset
        decom_y = self.dequantize_decom(decom_k)
        dct_y = decom_y if args.disable_subbands else get_blocks(decom_y, self.block_size, self.block_size)

        if args.perceptual_quantization:
            blocks_in_y = int(dct_y.shape[0]/self.block_size)
            blocks_in_x = int(dct_y.shape[1]/self.block_size)
            for by in range(blocks_in_y):
                for bx in range(blocks_in_x):
                    block = dct_y[by*self.block_size:(by+1)*self.block_size,
                                  bx*self.block_size:(bx+1)*self.block_size, :].astype(np.float32)
                    block[..., 0] /= (self.Y_QSSs/121)
                    block[..., 1] /= (self.C_QSSs/99)
                    block[..., 2] /= (self.C_QSSs/99)
                    dct_y[by*self.block_size:(by+1)*self.block_size,
                          bx*self.block_size:(bx+1)*self.block_size, :] = block

        ct_y = space_synthesize(dct_y, self.block_size, self.block_size)
        if self.use_lbt:
            ct_y = apply_lbt_2d(ct_y, self.block_size, inverse=True)
        ct_y = self.remove_padding(ct_y)
        y = to_RGB(ct_y) + self.offset
        y = np.clip(y, 0, 255).astype(np.uint8)
        output_size = self.decode_write_fn(y, out_fn)
        return output_size

    def decode(self, in_fn="/tmp/encoded", out_fn="/tmp/decoded.png"):
        return self.decode_fn(in_fn, out_fn)

    def quantize_decom(self, decom):
        logging.debug("trace")
        return self.quantize(decom)

    def dequantize_decom(self, decom_k):
        logging.debug("trace")
        return self.dequantize(decom_k)

if __name__ == "__main__":
    main.main(local_parser.parser, logging, CoDec)
