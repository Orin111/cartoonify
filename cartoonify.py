#################################################################
# FILE : cartoonify.py
# WRITER : orin pour , orin1 , 207377649
# EXERCISE : intro2cs2 ex5 2021
# DESCRIPTION:this file contain the cartoonify exercise
# STUDENTS I DISCUSSED THE EXERCISE WITH:
# WEB PAGES I USED:
# NOTES: ...
#################################################################
import math
import sys

import ex5_helper as eh


def separate_channels(image):
    """
    this function gets an image represented by a 3D list and return a list of
    x lists, each per channel
    :param image:a 3D list looks like this, image: [ rows: [ pixels: [R,G,B] ]]
    :return: a new list with a list of pixels per each channel
    example: if image= [[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]]
    the function will return:
    [[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]
    """
    # number of channels is the length of a pixel
    channels_num = len(image[0][0])
    # create a separate channels list with a list per each channel
    channels = [[] for i in range(channels_num)]
    for channel in range(channels_num):
        for row, row_lst in enumerate(image):
            new_row = []
            for column, pixel in enumerate(row_lst):
                # create a list of the channel per each row
                new_row.append(pixel[channel])
            # add the list to the specific channel
            channels[channel].append(new_row)
    return channels


def combine_channels(channels):
    """
    :param channels: image represented by list of x lists,
    each per channel
    :return: a new 3D list with a list of pixels
    example: if image=[[[1, 4], [7, 10]], [[2, 5], [8, 11]], [[3, 6], [9, 12]]]
    the function will return:
    [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
    """
    # create a separate list with a list per each row and column
    image = [[[] for j in range(len(channels[0][0]))]
             for i in range(len(channels[0]))]
    for row in range(len(channels[0])):
        for column in range(len(channels[0][0])):
            # create a separate rows list with a list per each pixel
            pixel = []
            for channel in range(len(channels)):
                pixel.append(channels[channel][row][column])
            image[row][column] = pixel
    return image


def RGB2grayscale(colored_image):
    """

    :param colored_image: a 3D colored image presented by list
    :return: a black and white image presented by 2D list
    """
    bw_image = [[[] for j in range(len(colored_image[0]))] for i in
                range(len(colored_image))]
    for row, row_lst in enumerate(colored_image):
        for column, pixel in enumerate(row_lst):
            # calculating the grey scale colored of each pixel
            bw_pixel = round(pixel[0] * 0.299 + pixel[1] * 0.587 +
                             pixel[2] * 0.114)
            bw_image[row][column] = bw_pixel
    return bw_image


def blur_kernel(size):
    """
    this function creating a kernel of a given size
    :param size: a size of a kernel
    :return a list in a size of the kernel **2
    """
    box_blur = []
    for i in range(size):
        box_blur.append([])
        for j in range(size):
            box_blur[i].append(1 / (size ** 2))
    return box_blur


def get_neighbors(row, column, k_size, image):
    """
    this function return the list of the pixel neighbors without pixel that
    outside of the image
    :param k_size: the size of the kernel (search is k_size 'radios')
    :param image: a list of an image
    :param row:the row of the pixel
    :param column:the column of the pixel
    :return: a list of the neighbors of the pixel
    """
    neighbors = []
    space = (k_size - 1) / 2
    for i in range(k_size):
        for j in range(k_size):
            x_dis = i - space
            y_dis = j - space
            if 0 <= x_dis + row < len(image) and 0 <= y_dis + column < len \
                        (image[0]):
                neighbors.append(image[int(x_dis + row)][int(y_dis + column)])
            else:
                neighbors.append(image[row][column])
    return neighbors


def apply_kernel(image, kernel):
    """
    :param image: an image presented by a 2D list
    :param kernel: a 2D list of a kernel
    :return:an image presented by a 2D list with the kernel calculation per
     each pixel
    """
    k_size = len(kernel)
    new_image = [[[] for j in range(len(image[0]))] for i in range(len(image))]
    for row in range(len(image)):
        for column in range(len(image[0])):
            neighbors = get_neighbors(row, column, k_size, image)
            sum_nk = 0
            for i in neighbors:
                # sum all pixel neighbors
                sum_nk += i / len(kernel) ** 2
            # if sum is bigger than 255> 255 if smaller than 0> 0
            sum_nk = round(min(max(0, sum_nk), 255))
            new_image[row][column] = sum_nk
    return new_image


def check_range(row, column, height, width):
    """ this function check if a pixel is in the range of the image"""
    if row >= height or row < 0:
        return False
    if column >= width or column < 0:
        return False
    return True


def bilinear_interpolation(image, y, x):
    """
    :param image: a 2D image
    :param y:row of the pixel in the image
    :param x:column of the pixel in the image
    :return: pixel value
    """
    column = math.floor(x)
    row = math.floor(y)
    a = image[row][column]
    if check_range(row + 1, column, len(image), len(image[0])):
        b = image[row + 1][column]
    else:
        b = a
    if check_range(row, column + 1, len(image), len(image[0])):
        c = image[row][column + 1]
    else:
        c = a
    if check_range(row + 1, column + 1, len(image), len(image[0])):
        d = image[row + 1][column + 1]
    else:
        d = b
    d_x = x - column
    d_y = y - row
    # pixel value calculation
    return round(
        (a * (1 - d_x) * (1 - d_y)) + (b * d_y * (1 - d_x)) + (c * d_x * (
                1 - d_y)) + (d * d_x * d_y))


def resize(image, new_height, new_width):
    """

    :param image: the source image, a 2D list
    :param new_height:the required height
    :param new_width:the required width
    :return:a new 2D image in the required height and width
    """
    new_image = [[0 for j in range(new_width)] for i in range(new_height)]
    h_dif = len(image) / new_height
    w_dif = len(image[0]) / new_width
    # updating the new image with the source values
    for row in range(new_height):
        for column in range(new_width):
            x = h_dif * row
            y = w_dif * column
            new_image[row][column] = bilinear_interpolation(image, x, y)
    # corners
    new_image[0][0] = image[0][0]
    new_image[new_height - 1][0] = image[len(image) - 1][0]
    new_image[0][new_width - 1] = image[0][len(image[0]) - 1]
    new_image[new_height - 1][new_width - 1] = image[len(image) - 1][
        len(image[0]) - 1]
    return new_image


def rotate_90(image, direction):
    """
    :param image: a 3D or 2D image presented by a list
    :param direction: the direction of the rotate 'L' for left 'R' for right
    :return a new image rotated to the requested direction
    """
    new_image = [[[] for j in range(len(image))] for i in range(len(image[0]))]
    for r, row in enumerate(image):
        for c, column in enumerate(row):
            if direction == 'R':
                new_image[c][len(image) - 1 - r] = column
            else:
                new_image[len(image[0]) - 1 - c][r] = column
    return new_image


def get_edges(image, blur_size, block_size, c):
    """
    :param image: a 2D image in B&W presented by a list
    :param blur_size: required blur
    :param block_size: required block of affect
    :param c: an int param, for the threshold calculation
    :return: a new image in B&W with only 0 or 255 pixels values
    """
    new_image = [[0 for j in range(len(image[0]))] for i in range(len(image))]
    blurred_image = apply_kernel(image, blur_kernel(blur_size))
    r = block_size // 2
    for i in range(len(new_image)):
        for j in range(len(new_image[0])):
            threshold = sum(get_neighbors(i, j, block_size, blurred_image)) / \
                        (block_size ** 2)
            if blurred_image[i][j] < threshold - c:
                new_image[i][j] = 0
            else:
                new_image[i][j] = 255

    return new_image


def quantize(image, N):
    """
    :param image: a 2D image
    :param N: number of shades
    :return: a new 2D image with the required number of shades
    """
    new_image = [[0 for j in range(len(image[0]))] for i in range(len(image))]
    for i in range(len(new_image)):
        for j in range(len(new_image[0])):
            new_image[i][j] = round(math.floor(image[i][j] * (N / 255)) *
                                    (255 / N))
    return new_image


def quantize_colored_image(image, N):
    """

    :param image: a 3D colored image
    :param N: number of shades
    :return: a new 3D image with the required number of shades
    """
    channels = len(image[0][0])
    new_image = [[] for i in range(channels)]
    sep_image = separate_channels(image)
    for c in range(channels):
        new_image[c] = quantize(sep_image[c], N)
    new_image = combine_channels(new_image)
    return new_image


def mask_2D(image1, image2, mask):
    """
    :param image1: a 2D image
    :param image2: a 2D image
    :param mask: a 2D image with only B&W pixels (255/ 0)
    :return: a combined image of all layers
    """
    new_image = [[0 for j in range(len(image1[0]))] for i in
                 range(len(image1))]
    for i in range(len(image1)):
        for j in range(len(image1[0])):
            new_image[i][j] = round(
                image1[i][j] * mask[i][j] + image2[i][j] * (1 - mask[i][j]))
    return new_image


def add_mask(image1, image2, mask):
    """
    :param image1: a 3D or 2D image
    :param image2: a 3D or 2D image
    :param mask: a 3D or 2D image with only B&W pixels (255/ 0)
    :return: a combined image of all layers
    """
    # if image is 3D
    if isinstance(image1[0][0], list):
        new_image = []
        sep_image1 = separate_channels(image1)
        sep_image2 = separate_channels(image2)
        for c in range(len(sep_image1)):
            new_image.append([mask_2D(sep_image1[c], sep_image2[c], mask)])
        new_image = combine_channels(new_image)
        # if image is 2D
    else:
        new_image = mask_2D(image1, image2, mask)
    return new_image


def create_mask(image):
    """
    :param image: a 2D image presented by a list with value of 0 or 255
    :return: a 2D image presented by a list with value of 0 or 1 (instead of
     255)
    """
    mask = [[[] for j in range(len(image[0]))] for i in range(len(image))]
    for row in range(len(image)):
        for column in range(len(image[0])):
            mask[row][column] = (image[row][column]) / 255
    return mask


def cartoonify(image, blur_size, th_block_size, th_c, quant_num_shades):
    """

    :param image: a 3D image
    :param blur_size: an int number> 0 , kernel size
    :param th_block_size: block size of pixels that will be affected
    :param th_c: parameter (int) that will determine the threshold
    :param quant_num_shades: number of shades that we will use for each color
    :return: a 3D cartooned image
    """
    edges_img = get_edges(RGB2grayscale(image), blur_size, th_block_size, th_c)
    mask = create_mask(edges_img)
    quantize_img = quantize_colored_image(image, quant_num_shades)
    sep_image = separate_channels(quantize_img)
    new_image = [[], [], []]
    for c in range(len(sep_image)):
        new_image[c] = add_mask(sep_image[c], edges_img, mask)
    return combine_channels(new_image)


def resize_colored(image, max_im_size):
    """
    :param image: a 3D image
    :param max_im_size:max height or width
    :return:a resized 3D image
    """
    highest = max(len(image), len(image[0]))
    size_def = highest / max_im_size
    sep_image = separate_channels(image)
    new_image = [[], [], []]
    for i in range(len(sep_image)):
        new_image[i] = (resize(sep_image[i], int(len(image) / size_def),
                               int(len(image[0]) / size_def)))
    return combine_channels(new_image)


def main():
    if len(sys.argv) == 8:
        # save all parameters
        image_source = sys.argv[1]
        cartoon_dest = sys.argv[2]
        max_im_size = int(sys.argv[3])
        blur_size = int(sys.argv[4])
        th_block_size = int(sys.argv[5])
        th_c = int(sys.argv[6])
        quant_num_shades = int(sys.argv[7])

        image = eh.load_image(image_source)
        re_image = resize_colored(image, max_im_size)
        eh.save_image(cartoonify(re_image, blur_size, th_block_size, th_c,
                                 quant_num_shades), cartoon_dest)

    else:
        print("Error: please enter all variables")


if __name__ == '__main__':
    main()
