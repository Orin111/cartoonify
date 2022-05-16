##############################################################################
# FILE: ex5_helper.py
# EXERCISE: Intro2cs ex5 2021-2022
# WRITER: Intro2CS 1 2021-2022 staff
# DESCRIPTION:A helper file for ex5 that masks handling with images
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from PIL import Image
from copy import deepcopy


##############################################################################
#                                 CONSTANTS                                  #
##############################################################################
GREYSCALE_CODE = "L"
RGB_CODE = "RGB"


##############################################################################
#                              Helper Functions                              #
##############################################################################
def load_image(image_filename):
    """
    Loads the image stored in the path image_filename and return it as a list
    of lists.
    :param image_filename: a path to an image file. If path doesn't exist an
    exception will be thrown.
    :return: a multi-dimensional list representing the image in the format
    rows X cols X channels. The list is 2D in case of a grayscale image and 3D
    in case it's colored.
    """
    img = Image.open(image_filename).convert('RGB')
    image = lists_from_pil_image(img)
    return image


def show_image(image):
    """
    Displays an image.
    :param image: an image represented as a multi-dimensional list of the
    format rows X cols X channels.
    """
    pil_image_from_lists(image).show()


def save_image(image, filename):
    """
    Converts an image represented as lists to an Image object and saves it as
    an image file at the path specified by filename.
    :param image: an image represented as a multi-dimensional list.
    :param filename: a path in which to save the image file. If the path is
    incorrect, an exception will be thrown.
    """
    pil_image_from_lists(image).save(filename)


def lists_from_pil_image(image):
    """
    Converts an Image object to an image represented as lists.
    :param image: a PIL Image object
    :return: the same image represented as multi-dimensional list.
    """
    width, height = image.size
    pixels = list(image.getdata())
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    if type(pixels[0][0]) == tuple:
        for i in range(height):
            for j in range(width):
                pixels[i][j] = list(pixels[i][j])
    return pixels


def pil_image_from_lists(image_as_lists):
    """
    Creates an Image object out of an image represented as lists.
    :param image_as_lists: an image represented as multi-dimensional list.
    :return: the same image as a PIL Image object.
    """
    image_as_lists_copy = deepcopy(image_as_lists)
    height = len(image_as_lists_copy)
    width = len(image_as_lists_copy[0])

    if type(image_as_lists_copy[0][0]) == list:
        for i in range(height):
            for j in range(width):
                image_as_lists_copy[i][j] = tuple(image_as_lists_copy[i][j])
        im = Image.new(RGB_CODE, (width, height))
    else:
        im = Image.new(GREYSCALE_CODE, (width, height))

    for i in range(width):
        for j in range(height):
            im.putpixel((i, j), image_as_lists_copy[j][i])
    return im


