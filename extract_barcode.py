import numpy as np
import cv2
import symImg1

# Here I define a method which takes a diagonally symmetric image and extracts the barcode from it
# We represent barcodes as a collection of intervals [(E_b,E_d)...], where each interval represents
# a topologial feature appears at the filtration value Epsilon_b and vanishes at filtration value Epsilon_d

# Another observation is that once a connected component appears in the barcode, it will never disappear
# This means that we only need to keep track of the birth of connected components




# The method should return two lists, one for the barcode in the bottom left corner and one for the barcode in the top right corner
def extract_barcode(symmetric_image):
    bl_barcode = []
    tr_barcode = []





    return bl_barcode, tr_barcode