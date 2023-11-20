# It is a class to hold detectors of the CSST MCI for dark current and flat field modelling
# Initial format from PengJia TYUT for the CSST MCI Detectors
import numpy as np
import datetime
from scipy.ndimage import generic_filter
import os
from astropy.io import fits
import re
import glob
import SEx1

class Chips:
    def __init__(self, chip_id, chip_config=None, configfile=None, configupdate=False):
        """
        :param chip_id: Type: String
        :param chip_config: Type: Dictionary
        :param configfile: Type: Dictionary, contains 'dark' 'bad' and 'flat'
        :param configupdate: Type: Flag to make sure whether to save in this time
        The chip_id stands for the id of the chips, which could be used to load pre-processed data
        The chip_config stands for some configurations, which stand for the working conditions/observation dates etc
        Define necessary parameters that load from the hard disk
        """
        self.chip_id = chip_id
        if chip_config is not None:
            self.chip_config = chip_config
        else:
            # Load default chip configurations
            chip_config = {
                'Status': 'G',
                'DarkCurrent': None,
                'BadPixel': None,
                'year': '2023',
                'FlatField': None,
                'size': [4936, 23984],
                'Temperature': 0,
                'ExpTime': 300,
                'Gain': 1.55,
                'bin': 1,
                'BiasVol': 500.0
            }
            self.chip_config = chip_config
        if configfile is None:
            matsize = self.chip_config['size']
            # Generate configuration file
            self.darkcurrentmat = np.zeros(matsize)
            self.badpixelmat = np.zeros(matsize)
            self.flatmat = np.zeros(matsize)
        else:
            self.darkcurrentmat = configfile['dark']
            self.badpixelmat = configfile['bad']
            self.flatmat = configfile['flat']
        if configupdate is True:
            # We save configure files here
            self.saveflag = True
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        print(str( datetime.datetime.now()),'\n')
        print('Autobots, transform and roll out !\n')

    def remove_abnormal_point(self,image,mask=None):
        '''
        :param image: The image as a matrix.                Type:Numpy.
        :param mask: The mask matrix of the image.   Type:Numpy or None
        call default bad pixel mask if necessary
        :return: image: Remove the anomaly picture.  Type:Numpy.
        '''
        #Load Bad pixel mat as default
        if mask is None:
            mask = self.badpixelmat
        # mask becomes the mask matrix of True and False
        mask = np.isnan(mask)
        # Create a temporary image matrix
        masked_image = np.copy(image).astype(float)
        # Set the outlier on the temporary image matrix to NAN
        masked_image[mask] = np.nan
        # Create a 9*9 average filter
        kernel = np.ones((9, 9)) / 81
        # The NaN value in the mask is averaged by nearby pixel values
        filtered_image = generic_filter(masked_image, mean_filter,
                                    size=kernel.shape, mode='constant', cval=np.NaN)
         # Copy the filtered value back to the original image
        image[mask] = filtered_image[mask]
        return image

    # Defines a function for applying an average filter at a given position
    def mean_filter(x):
        # Replace NaN with 0 and calculate the mean
        x = np.nan_to_num(x, nan=0.0)
        return np.mean(x)

    def save_new_fits(img, dir, name):
        """
        Purpose: Saves a new FITS image file to the specified directory with the given name.

        :param img: The FITS image to be saved. Type: FITS Image.
        :param dir:  The directory where the FITS image will be saved. Type: String.
        :param name:    The name of the FITS image . Type: String.
        """
        path = os.path.join(dir, name)
        if os.path.exists(path):
            os.remove(path)
        grev = fits.PrimaryHDU(img)
        grevHDU = fits.HDUList([grev])
        grevHDU.writeto(path)
        print("save fits:{}".format(path))
        #OK We have done
        return 0

    # Obtain star coordinates via the sextractor
    def Generate_Celestial_Coordinates(fits):
        #TODO: A lot of refactor needs to be done for zhimin's code PENG 20231101
        keys = ['DETECT_TYPE',
                'DETECT_MINAREA',
                'DETECT_THRESH',
                'ANALYSIS_THRESH',
                'DEBLEND_NTHRESH',
                'DEBLEND_MINCONT',
                'CLEAN_PARAM',
                'BACK_SIZE',
                'BACK_FILTERSIZE']
        # Set corresponding parameters
        values = []
        #TODO: Why we need root to run SExtractor, working under docker? PENG 20231101
        sex = SEx1.ColdHotdetector(
            fits, './sextractor/image2.sex',
            './sextractor/simplify.sex', './sextractor/test.param',
            True, '741236985', keys, values)
        result = sex.cold
        #OK there is a way called warm and cold extraction, modify it later...
        result = [i[:2] for i in result]
        result = np.array(result, dtype='int_')
        return result

    # Generate the mask matrix
    def Generate_mask(mask, Galactic_coordinates):
        #Modified by Peng: The Galactic_coordinates are not good here. We are actually assigning masks to detector,
        # not necessarly to use such a name PENG20231101
        background_mask = np.copy(mask).astype(float)
        background_mask[background_mask >= 0] = 1
        background_mask[np.isnan(background_mask)] = 0
        for coord in Galactic_coordinates:
            #TODO: Why we need to mask 10 pixels? No good Peng 20231101
            background_mask[coord[0] - 5:coord[0] + 5, coord[1] + 5:coord[1] - 5] = 0
        return background_mask

    # Select a dark frame with similar background levels
    def Select_similar_backgrounds(self, background_mask, image):
        #Folder and files to save all these files are not necessary now, path, interval):
        #Modififed by Peng, now we have id and dark current frame from configfile,
        # which are used to store all dark current frames by PENG20231101 the following code is not necessary
        #dark_background_arr = []
        # Get the number of dark field images
        #img_num = len(os.listdir(path))
        # Initializes a dark field three-dimensional matrix
        #cube = np.zeros((2048, 2048, img_num))
        # Defining regular expressions
        #pattern = r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})"
        # Gets the file name and time information for all files in the folder
        #files = [(f, re.findall(pattern, f)[0]) for f in os.listdir(path) if re.findall(pattern, f)]
        # Sort files by time information
        #sorted_files = sorted(files, key=lambda x: x[1])
        #sorted_files = np.array(sorted_files)
        # Calculate the background mean of the image to be processed
        #First estimate the background level
        if len(self.darkcurrentmat) == 2:#They are pure 2D matrix here
            dark_image = self.darkcurrentmat

        elif len(self.darkcurrentmat) == 3:#They are in our direction
            cube = self.darkcurrentmat
            image_background = np.multiply(background_mask, image).mean()
            dark_background_list = np.multiply(background_mask, dark_image).mean((1,2)) #No iteration is required,
            # just broadcasting....
            min_idx = np.argmin(np.abs(dark_background_list-image_background))
            # The two-dimensional matrix with the closest dark field is obtained
            dark_image = cube[:, :, min_idx]
        """
        #Not necessarly useful here
        # Traverse the background mean of the dark field image
        for i, filename in enumerate(sorted_files[:, 0]):
            dark_image = fits.open(path + '/' + filename)[0].data
            # The dark field matrix is obtained by traversing
            cube[:, :, i] = dark_image
            # Get the background mean of the dark field image
            dark_background = np.multiply(background_mask, dark_image).mean()
        # Calculate which dark field image is closest to the image background mean
        dark_background_arr = np.array(dark_background_arr)
        dark_background_arr = abs(image_background - dark_background_arr)
        min_idx = np.argmin(dark_background_arr)
        # Calculate the exposure time of images with similar dark fields
        Exposure_time = min_idx * interval
        # The two-dimensional matrix with the closest dark field is obtained
        dark_image = cube[:, :, min_idx]
        """
        #return Exposure_time, dark_image
        return dark_image

    # Calculated dark current with Polynomials
    def Calculate_Dark_Current(Polynomial, Clucster, Exposure_time):
        # Obtain the polynomial coefficients of the corresponding clusters
        coefficient = Polynomial[Clucster]
        Dark = []
        return Dark

    # Generates a dark current matrix
    def Generate_Dark_Matrix(mask, Polynomial, Exposure_time, dark_image):
        # Create a temporary mask matrix
        temporary_mask = np.copy(mask)
        # Go through each cluster, calculate the dark current, and assign the value to the mask matrix
        for i in range(Polynomial.shape[0]):
            Dark = Calculate_Dark_Current(Polynomial, i, Exposure_time)
            temporary_mask[temporary_mask == i] = Dark
        temporary_mask[np.isnan(mask)] = 0
        temporary_mask = temporary_mask - dark_image
        return temporary_mask

    # Subtracting dark current
    def subtract_dark(dark_matrix, image):
        image = abs(image - dark_matrix)
        return image

    def preprocessing(path, mask, dark_path, Polynomial, save_path):
        for filename in os.listdir(path):
            fitPath = path + "/" + filename
            image = fits.open(path + '/' + filename)[0].data
            image = remove_abnormal_point(image, mask)
            Galactic_coordinates = Generate_Celestial_Coordinates(fitPath)
            backgrounds_mask = Generate_mask(mask, Galactic_coordinates)
            expouse_time, dark_image = Select_similar_backgrounds(backgrounds_mask, image, dark_path, interval=1)
            dark_matrxi = Generate_Dark_Matrix(mask, Polynomial, expouse_time, dark_image)
            image = subtract_dark(dark_matrxi, image)
            save_new_fits(image, save_path, filename)
            print('{} completed'.format(filename))