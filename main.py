# This is the start-up for p
import numpy as np
import scipy
from Detector import Chips
import datetime
import numpy as np
from astropy.io import fits
import os
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn import mixture
import logging
import re

def run_procedure():
    virdetector = Chips('CSST_MCI')

logging.basicConfig(filename='result.log', level=logging.DEBUG)


def modeling_init(datapath):
    # Determine picture size
    size_m = input('Please enter the picture size m:')
    size_n = input('Please enter the picture size n:')
    size_m = int(size_m)
    size_n = int(size_n)
    print('Image size:', size_m, size_n)
    index = np.arange(size_m * size_n).astype(float)
    # Gets the number of fits images
    img_num = len(os.listdir(datapath))
    print(type(img_num))
    print('img_num:{}'.format(img_num))
    # Initializes the time series matrix
    time_series_cube = np.zeros((size_m, size_n, img_num))
    # Defining regular expressions
    pattern = r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})"
    # Gets the file name and time information for all files in the folder
    files = [(f, re.findall(pattern, f)[0]) for f in os.listdir(datapath) if re.findall(pattern, f)]
    # Sort files by time information
    sorted_files = sorted(files, key=lambda x: x[1])
    sorted_files = np.array(sorted_files)
    # 创建时间序列矩阵
    for i, filename in enumerate(sorted_files[:, 0]):
        print(filename)
        img = fits.open(datapath + '/' + filename)
        img = img[0].data
        time_series_cube[:, :, i] = img
    print('Dimension of 3D modeling matrix：', time_series_cube.shape)
    # Dimensionality reduction
    DR_time_series_cube = time_series_cube.reshape(-1, img_num)
    print('Reduce the dimension array size:{}'.format(DR_time_series_cube.shape))
    return time_series_cube, DR_time_series_cube, index, size_m, size_n, img_num


# Save data file
def save_data(mask, f1_arr):
    now = str(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    mask_name = now + "mask.npy"
    f1_arr_name = now + 'f1_arr.npy'
    np.savetxt(mask_name, mask)
    np.savetxt(f1_arr_name, f1_arr)


# The optimal number of clusters is obtained
def get_optimal_number(X):
    # Define the range of cluster numbers
    n_components_range = range(1, 30)

    # Calculate the BIC value for the number of clusters
    bics = []
    for n_components in n_components_range:
        model = mixture.GaussianMixture(n_components, covariance_type='full')
        model.fit(X)
        bics.append(model.bic(X))
        logging.info("The cluster{} bic {} ".format(n_components, model.bic(X)))

    # Draw BIC curve
    plt.plot(n_components_range, bics)
    plt.xlabel('Number of Components')
    plt.ylabel('BIC')
    plt.title('BIC for Gaussian Mixture Model')
    plt.savefig('bic_plot.png', format='png')


def log_model(x, a, b, c):
    return a * np.log(b * x) + c


def gauss_clustering(DR_time_series_cube, time_series_cube, clusters_number, img_num, size_m, size_n, index, n,
                     DR_time_series_cube_latest=None, true_mask=None):
    # 初始化
    f1_arr = []
    outliner_arr = []

    # cluster
    if n == 0:
        gmm = mixture.GaussianMixture(covariance_type='full', n_components=clusters_number)
        gmm.fit(DR_time_series_cube)
        clusters = gmm.predict(DR_time_series_cube)
        print('Clustering completion')
    else:
        gmm = mixture.GaussianMixture(covariance_type='full', n_components=clusters_number)
        gmm.fit(DR_time_series_cube_latest)
        clusters = gmm.predict(DR_time_series_cube_latest)
        print('Clustering completion')

    if n == 0:
        mask = clusters.reshape(size_m, size_n).astype(float)
    else:
        mask = true_mask.astype(float)

    if np.isnan(mask).any():
        # A counter that records the number of processed values in a one-dimensional array
        count = 0

        # Traverse the two-dimensional array line by line
        for i in range(mask.shape[0]):
            # Traverse the two-dimensional array column by column
            for j in range(mask.shape[1]):
                # If the current position is not assigned, take a value from the one-dimensional array and assign it
                if mask[i, j] is None:
                    # If the one-dimensional array has been processed, exit the loop
                    if count == clusters.shape[0]:
                        break
                    # Otherwise, a value is assigned to the current position and the counter is increased by 1
                    else:
                        mask[i, j] = clusters[count]
                        count += 1
    # Each cluster is traversed
    for i in range(clusters_number):
        # Find all coordinates belonging to cluster i in mask
        rows, cols = np.where(mask == i)
        rows_len = len(rows)
        # Initialize the time vector of cluster i
        clusters_cube = np.zeros((rows_len, img_num))
        print('clusters_cube shape{}'.format(clusters_cube.shape))
        # For each kind of cluster, the corresponding time vector is found in the three-dimensional time series, and they are combined into a two-dimensional vector
        for j in range(rows_len):
            clusters_cube[j] = time_series_cube[rows[j], cols[j], :]
        # Define time series coordinates
        time_series = np.arange(1, img_num)
        # Calculate the reciprocal of the temperature
        inverse_temperature = 1 / time_series
        # print(len(inverse_temperature))
        # Average the time vector of a cluster as the Y-axis
        y = clusters_cube.mean(axis=0)
        # Logarithmic transformation
        log_dark_current = np.log(y)
        params, covariance = curve_fit(log_model, inverse_temperature, log_dark_current, p0=[1, 1, 1])
        # Preservation factor
        f1_arr.append(params)
        # Fits the y value
        yvals = log_model(time_series, *params)
        # Calculate the mean of the residual difference between each time vector and the predicted value of the same cluster
        clusters_mean = np.abs(np.mean(yvals - np.log(clusters_cube), axis=1))
        # Calculate the standard deviation of each time vector and prediction residual for the same cluster
        clusters_std = np.std(yvals - np.log(clusters_cube), axis=1)
        # Calculate the degree of fit MSE of a cluster
        clusters_MSE = np.mean(np.square(yvals - np.log(clusters_cube)))
        logging.info("The{} cluster{} MSE {} ".format(n, i, clusters_MSE))
        # To judge the anomaly, the criterion is that the residual variance is greater than n times of the mean
        outliner = np.stack((rows[np.where(clusters_std > n * clusters_mean)],
                             cols[np.where(clusters_std > n * clusters_mean)]), axis=1)
        # Save outlier
        outliner_arr.append(outliner)
        # The outlier is set to nan on the mask matrix
        for x, y in outliner:
            mask[x, y] = np.nan
        # Reconstructing the dimensionality reduction time series matrix, the time series corresponding to the anomaly point is not brought into the next cluster
        outliner_index = np.array([x * 2048 + y for x, y in outliner])
        print(outliner_index)
        if len(outliner_index) != 0:
            index[outliner_index] = np.nan
            DR_time_series_cube_latest = DR_time_series_cube[~np.isnan(index)]
            logging.info("Reduce the dimension array size:{} ".format(DR_time_series_cube_latest.shape))
    n = n + 1
    if n < 3:
        save_data(mask, f1_arr)
        gauss_clustering(DR_time_series_cube, time_series_cube, clusters_number, img_num, size_m, size_n, index, n,
                         DR_time_series_cube_latest, mask)
    return mask, DR_time_series_cube, DR_time_series_cube_latest


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_procedure()

