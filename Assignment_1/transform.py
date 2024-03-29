import numpy as np
from skimage import io
import os.path as osp
import math as math

def load_image(file_name):
    """
    Load image from disk
    :param file_name:
    :return: image: numpy.ndarray
    """
    if not osp.exists(file_name):
        print('{} not exist'.format(file_name))
        return
    image = io.imread(file_name)
    return np.array(image)

def save_image(image, file_name):
    """
    Save image to disk
    :param image: numpy.ndarray
    :param file_name:
    :return:
    """
    io.imsave(file_name,image)

def cs4243_resize(image, new_width, new_height):
    """
    10 points
    Implement the algorithm of nearest neighbor interpolation for image resize,
    :param image: ndarray
    :param new_width: int
    :param new_height: int
    :return: new_image: numpy.ndarray
    """
    new_image = np.zeros((new_height, new_width, 3), dtype='uint8')
    if len(image.shape)==2:
        new_image = np.zeros((new_height, new_width), dtype='uint8')
    ###Your code here####

    height_ratio = image.shape[0] / new_height
    width_ratio = image.shape[1] / new_width

    for row_number in range(new_image.shape[0]):
        for col_number in range(new_image.shape[1]):
            old_col = int(col_number * width_ratio)
            old_row = int(row_number * height_ratio)
            new_image[row_number][col_number] = image[old_row][old_col]

    ###
    return new_image

def cs4243_rgb2grey(image):
    """
    5 points
    Implement the rgb2grey function
    weights for different channel: (R,G,B)=(0.299, 0.587, 0.114)
    Please scale the value to [0,1] by dividing 255
    :param image: numpy.ndarray
    :return: grey_image: numpy.ndarray
    """
    if len(image.shape) != 3:
        print('Image should have 3 channels')
        return
    ###Your code here####
    new_image = np.zeros((image.shape[0], image.shape[1]), dtype='float')
    weights = np.array([0.299, 0.587, 0.114])

    for row_number in range(image.shape[0]):
        for col_number in range(image.shape[1]):
            new_image[row_number][col_number] = np.dot(weights, image[row_number][col_number])

    return new_image/255.
    ###

def cs4243_rotate180(kernel):
    """
     5 points
    Rotate the matrix by 180
    :param kernel:
    :return:
    """
    ###Your code here####
    kernel = np.rot90(kernel)
    kernel = np.rot90(kernel)

    ###
    return kernel

def cs4243_guassian_kernel(ksize, sigma):
    """
     10 points
    Implement the simplified Guassian kernel below:
    k(x,y)=exp((x^2+y^2)/(-2sigma^2))
	Note that Guassian kernel should be central symmentry.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    """
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    ###Your code here####
    half_ksize = int(ksize / 2)
    if ksize % 2 == 0:
        half_ksize -= 1

    for row in range(ksize):
        for col in range(ksize):
            row_adjusted = half_ksize - row
            col_adjusted = half_ksize - col
            exp_val = np.exp((np.square(row_adjusted) + np.square(col_adjusted))/(-2 * np.square(sigma)))
            kernel[row][col] = exp_val

    return kernel / kernel.sum()

def cs4243_filter(image, kernel):
    """
    15 points
    Implement the convolution operation in a naive 4 nested for-loops,
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return:
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####

    kernel = cs4243_rotate180(kernel)

    pad_height = int((Hk - 1) / 2)
    pad_width = int((Wk - 1) / 2)
    image = pad_zeros(image, pad_height, pad_width)

    half_Hk = int(Hk/2)
    half_Wk = int(Wk/2)

    for Ri in range(pad_height, Hi + pad_height):
        for Ci in range(pad_width, Wi + pad_width):
            val = 0
            for Rk in range(Hk):
                for Ck in range(Wk):
                    H_from_centre = half_Hk - Rk
                    W_from_centre = half_Wk - Ck
                    val += image[Ri - H_from_centre][Ci - W_from_centre] * kernel[Rk][Ck]
            filtered_image[Ri - pad_height][Ci - pad_width] = val

    ###

    return filtered_image

def pad_zeros(image, pad_height, pad_width):
    """
    Pad the image with zero pixels, e.g., given matrix [[1]] with pad_height=1 and pad_width=2, obtains:
    [[0 0 0 0 0]
    [0 0 1 0 0]
    [0 0 0 0 0]]
    :param image: numpy.ndarray
    :param pad_height: int
    :param pad_width: int
    :return padded_image: numpy.ndarray
    """
    height, width = image.shape
    new_height, new_width = height+pad_height*2, width+pad_width*2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height-pad_height, pad_width:new_width-pad_width] = image
    return padded_image

def cs4243_filter_fast(image, kernel):
    """
    20 points
    Implement a fast version of filtering algorithm.
    Do element-wise multiplication between
    the kernel and a image region.
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####

    kernel = cs4243_rotate180(kernel)

    pad_height = int(np.rint((Hk - 1) / 2))
    pad_width = int(np.rint((Wk - 1) / 2))
    image = pad_zeros(image, pad_height, pad_width)

    for Ri in range(image.shape[0] - pad_height - 1):
        for Ci in range(image.shape[1] - pad_width - 1):
            image_section = image[Ri:Ri + Hk, Ci:Ci + Wk]
            val = np.multiply(image_section, kernel).sum()
            filtered_image[Ri][Ci] = val
    ###

    return filtered_image

def cs4243_filter_faster(image, kernel):
    """
    25 points
    Implement a faster version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and arrange them into a matrix of shape (Hi*Wi, Hk*Wk),also arrage the flipped
    kernel to be of shape (Hk*Hk, 1), then do matrix multiplication\
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    filtered_image = np.zeros((Hi, Wi))

    ###Your code here####

    kernel = cs4243_rotate180(kernel)

    pad_height = int(np.rint((Hk - 1) / 2))
    pad_width = int(np.rint((Wk - 1) / 2))
    image = pad_zeros(image, pad_height, pad_width)

    Ht = Hk*Wk
    Wt = Hi*Wi
    input_transformed = np.zeros((Ht, Wt))
    new_kernel = kernel.reshape(1, (Hk * Wk))

    for Ri in range(Hi):
        for Ci in range(Wi):
            image_section = image[Ri:Ri + Hk, Ci:Ci + Wk]
            input_transformed[:Ht, Ci + Ri*Wi] = image_section.reshape(Ht)

    multiplied = np.dot(new_kernel, input_transformed)
    filtered_image = multiplied.reshape(Hi, Wi)

    ###

    return filtered_image

def cs4243_downsample(image, ratio):
    """
    10 points
    Downsample the image to its 1/(ratio^2),which means downsample the width to 1/ratio, and the height 1/ratio.
    for example:
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    B = downsample(A, 2)
    B=[[1, 3], [7, 9]]
    :param image:numpy.ndarray
    :param ratio:int
    :return:
    """

    ###Your code here####
    new_width = int(np.rint(image.shape[1]/ratio))
    new_height = int(np.rint(image.shape[0]/ratio))
    downsample_image = np.zeros((new_height, new_width))
    for row_number in range(downsample_image.shape[0]):
        for col_number in range(downsample_image.shape[1]):
            old_col = col_number * ratio
            old_row = row_number * ratio
            downsample_image[row_number][col_number] = image[old_row][old_col]
    ###
    return downsample_image