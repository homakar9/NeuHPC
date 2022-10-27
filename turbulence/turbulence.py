#!/usr/bin/python3

import os
import re
import sys
import glob
import math
import shutil
import pickle
import tempfile
import itertools
import collections
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt

import cv2

# shapely geometry to turn point cloud into polygon and compute convex hull.
import shapely
import shapely.geometry
from shapely.geometry import Point, Polygon, LineString, MultiPolygon

from collections import Counter
from sklearn.cluster import DBSCAN



def rgbgr(img):
    """ Turns BGR (opencv) to RGB (everything else) and vice versa. """
    img2 = img.copy()
    img2[:, :, 0] = img[:, :, 2]
    img2[:, :, 2] = img[:, :, 0]
    return img2

def cluster_to_image(cluster):
    new_img = np.zeros_like(img)
    xs = cluster[:,0]
    ys = cluster[:,1]
    for x,y in zip(xs,ys):
        new_img[y,x] = 255
    return new_img

def clusters_to_image(clusters, image_shape):
    merged_img = np.zeros(image_shape)
    for n, cluster in clusters.items():
        xs = cluster[:,0]
        ys = cluster[:,1]
        for x,y in zip(xs,ys):
            merged_img[y,x] = 255
    return merged_img

def eliminate_noise_clusters(clusters, min_points_threshold = 300):
    real_clusters = {}
    for n, cluster in clusters.items():
        if n == -1: continue
        if len(cluster) < min_points_threshold: continue
        real_clusters[n] = cluster
    return real_clusters

def html_color_to_tuple(color):
    color = color.strip('#')
    channels = [color[0:2], color[2:4], color[4:]]
    channels = tuple(eval(f"0x{byte}") for byte in channels)
    return channels

def iter_colors():
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for color in colors:
        color = html_color_to_tuple(color)
        deltas = [np.random.uniform(-60, 60) for _ in range(3)]
        color = tuple(np.clip([*map(sum, zip(color, deltas))], 0, 255))
        yield color


# Define current sheet object

class CurrentSheet:

    def __init__(self, id, points, color = (255, 255, 255)):
        self.id = id
        self.points = points
        self.color = color

    @property
    def xs(self):
        return self.points[:,0]

    @property
    def ys(self):
        return self.points[:,1]

    @property
    def polygon(self):
        return Polygon(self.points)

    @property
    def length(self):
        return self.polygon.convex_hull.length


    def concave_hull(self, k = 1):
        from hulls import ConcaveHull
        print(f"Begin computing concave hull for {self}")
        concave_hull = ConcaveHull(self.points)
        concave_hull_points = concave_hull.calculate(k = k)
        print(f"Finished computing concave hull for {self}")
        return concave_hull_points

    def concave_hull_circumference(self, k = 1):
        concave_hull_points = self.concave_hull(k = k)
        distance = 0
        for pair1, pair2 in zip(concave_hull_points[:-1], concave_hull_points[1:]):
            x1, y1 = pair1
            x2, y2 = pair2
            dx = (x2 - x1)
            dy = (y2 - y1)
            distance += np.sqrt(dx**2 + dy**2)
        return distance


    def blit(self, img_array = None, img_shape = None):

        if img_array is None and img_shape is None:
            raise ValueError(f"Must specify one of: img_array or img_shape")

        if img_array is None:
            img_array = np.zeros([*img_shape, 3])

        for x, y in zip(self.xs, self.ys):
            img_array[y, x, 0] = int(self.color[0])
            img_array[y, x, 1] = int(self.color[1])
            img_array[y, x, 2] = int(self.color[2])

        return img_array


def get_unique_colors(n):
    colors = {}
    color_generator = iter_colors()
    for n in range(n):
        colors[n] = next(color_generator)
    return colors


def read_gda_file(file, shape):
    """
    :param file: The gda file
    :return: numpy array of extracted file
    """
    with open(file, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)

    np_data = np.reshape(data, shape, order='F')
    return np_data[::-1, :]


def load_frame(path, brighten_factor = 2, plot = False):

    if path.endswith('.gda'):
        gda_shape = (8*1024, 8*2048)
        array = read_gda_file(path, gda_shape)
    elif path.endswith('.png'):
        array = cv2.imread(path, 0)
    else:
        raise TypeError('No')

    array = cv2.resize(array, (2048, 1024))
    array = array[10:-10] # chop off the top and bottom 10 pixels to avoid the weird line effect.

    #array = np.sqrt(array)
    array = np.square(array)
    array = cv2.GaussianBlur(array, ksize = (5, 5), sigmaX = 2)
    #array = cv2.GaussianBlur(array, ksize = (15, 15), sigmaX = 5)
    array = (array - array.min())
    array = (array / array.max())
    array = (255*array)

    #if brighten_factor is not None:
    #    array = brighten_factor*(array-array.min())
    #    array = np.clip(array, 0, 255)

    #array[array < 200] = 0
    #array[array > 0] = 255

    array = array.astype(np.uint8)

    #threshold = 222
    #top_quantile_or_threshold = max(threshold, int(np.quantile(array, 0.99)))
    lower_threshold = int(np.quantile(array, 0.97))
    #lower_threshold = 222
    #thresh, img = cv2.threshold(array, lower_threshold, 255, cv2.THRESH_BINARY)
    array = cv2.inRange(array, lower_threshold, 255)

    if plot:
        plt.imshow(array, cmap = 'afmhot')
        plt.show()


    return array


def get_number_from_gda_pathname(gda_path):
    m = re.match(f"(.*absJ_)([0-9]+)([.]gda)", gda_path)
    return int(m.groups()[1])


def solve_turbulence(image_num, image_path, previous_current_sheets = None):

    basename = os.path.basename(image_path)
    stem, ext = os.path.splitext(basename)
    output_path = f'{OUTPUT_DIR}/{image_num:05d}.png'

    if os.path.exists(output_path):
        print(f"Already have image: {output_path}. Returning.")
        return []

    print(f"{image_num:05d}: Beginning run for {image_path}")

    img = load_frame(image_path)

    #cv2.imwrite(output_path, img_raw)
    #return []

    # threshold image

    if False:

        tmp_dir = tempfile.mkdtemp()

        shutil.copy(image_path, f"{tmp_dir}/before.png")
        # old way involving copying the image to RAM in /tmp
        cv2.imwrite(f"{tmp_dir}/before.png", img_raw)
        os.system(f"convert -threshold 90% '{tmp_dir}/before.png' '{tmp_dir}/after.png'")

        # load image
        img = cv2.imread(f'{tmp_dir}/after.png', 0)

        img_shape = img.shape

        shutil.rmtree(tmp_dir)


    img_shape = img.shape

    # edge detection using canny algorithm
    edges = cv2.Canny(img, 0, 255)

    # nonzero pixels in canny output give point cloud
    ys, xs = np.where(edges != 0)
    point_cloud = np.vstack((xs,ys)).T

    USE_TDA = False

    if USE_TDA:

        import tda

        def cluster(data):

            tomato = tda.ToMaTo(data)
            persistence = tomato.estimate_clusters()

            #tomato.plot_clusters(persistence)

            cluster_lifetimes = tomato.get_cluster_lifetimes(persistence)
            #real_cluster_lifetimes = tomato.barcode_to_num_clusters(cluster_lifetimes, threshold = 2)

            # TODO: Gaussian mixture model? Kmeans? Here we *know* we only have two classes,
            # since it's just "stuff tethered to the max value" and "junk", so this isn't
            # an infinite regress of lols.

            #hacky_cluster_lifetimes = [x for x in cluster_lifetimes if np.log(x) > -15.6]
            #cluster_lifetimes = hacky_cluster_lifetimes

            num_clusters = len(cluster_lifetimes)
            something = tomato.fit_predict(num_clusters)
            clustered_index_arrays = tomato.sts

            clusters = dict(enumerate([data[i] for i in clustered_index_arrays]))

            return clusters

        clusters = cluster(point_cloud)

    else:

        # dbscan to get clusters.

        eps = 5
        dbscan = DBSCAN(eps = eps, min_samples = 1)
        try:
            cluster_indexes = dbscan.fit_predict(point_cloud)
        except:
            # save dummy image
            main_image = np.zeros([*img_shape, 3])
            cv2.imwrite(output_path, main_image)
            return


        clusters = {n : point_cloud[cluster_indexes == n] for n in set(cluster_indexes)}

    ###################

    real_clusters = eliminate_noise_clusters(clusters, min_points_threshold = 100)
    merged_image = clusters_to_image(real_clusters, img_shape)

    # color clusters to make colored image

    real_cluster_pairs_sorted_by_sizes = sorted(
        real_clusters.items(),
        key = lambda pair: len(pair[1]),
        reverse = True,
    )

    real_cluster_pairs_sorted_by_sizes = [p[1] for p in real_cluster_pairs_sorted_by_sizes]

    def get_features_for_cluster(points):
        features = []
        for column in (0, 1):
            for function in (np.mean, np.min, np.max):
                features.append(function(points[:, column]))
        #features.append(points.shape[0])
        return np.array(features)

    def get_cluster_distance(old_points, new_points):
        old_features = get_features_for_cluster(old_points)
        new_features = get_features_for_cluster(new_points)
        distance = np.linalg.norm(old_features - new_features)
        return distance

    def get_best_match(points, current_sheet_list):
        ranks = {cs.id: get_cluster_distance(new_points, cs.points) for cs in previous_current_sheets}
        id, rank = sorted(ranks.items(), key = lambda pair: pair[1])[0]
        best_matched_current_sheet = previous_current_sheets[id]
        return best_matched_current_sheet

    if False:

        affinities = {}
        for n_new, new_points in enumerate(real_cluster_pairs_sorted_by_sizes):
            for cs in previous_current_sheets:
                affinities[(n_new, cs.id)] = get_cluster_distance(new_points, cs.points)

        matches = []
        for n_new, new_points in enumerate(real_cluster_pairs_sorted_by_sizes):
            for cs in previous_current_sheets:
                ranks_for_left  = {pair:value for pair,value in affinities.items() if pair[0] == n_new}
                ranks_for_right = {pair:value for pair,value in affinities.items() if pair[1] == cs.id}
                ranks_for_left = sorted(ranks_for_left.items(), key = lambda pair: pair[1])
                ranks_for_right = sorted(ranks_for_right.items(), key = lambda pair: pair[1])
                #if (ranks_for_left[0][0] == ranks_for_right[0][0][::-1]):
                #    print(ranks_for_left[0][0], ranks_for_right[0][0])
                matches.append(ranks_for_left[0][0])
                matches.append(ranks_for_right[0][0])

        matches = sorted(set(matches))
        from collections import defaultdict
        ownership = defaultdict(list)
        for new, old in matches:
            ownership[old].append(new)


    current_sheets = []

    if previous_current_sheets:
        matched_clusters_from_previous_generation = set()

        for n_new, new_points in enumerate(real_cluster_pairs_sorted_by_sizes):

            bmcs = get_best_match(new_points, previous_current_sheets)

            if bmcs.id in matched_clusters_from_previous_generation:
                current_sheet = CurrentSheet(id = n_new, points = new_points, color = colors[n_new])
            else:
                color = bmcs.color
                current_sheet = CurrentSheet(id = n_new, points = new_points, color = color)
                matched_clusters_from_previous_generation.add(bmcs.id)

            current_sheets.append(current_sheet)

    else:

        for n, cluster in enumerate(real_cluster_pairs_sorted_by_sizes):
            current_sheet = CurrentSheet(id = n, points = cluster, color = colors[n])
            current_sheets.append(current_sheet)


    colored_img = np.zeros([*img_shape, 3])

    for current_sheet in current_sheets:
        current_sheet.blit(img_array = colored_img)

    colored_img = colored_img.astype(np.uint8)

    # cv2.imwrite(f'{OUTPUT_DIR}/{n:05d}.png', colored_img)

    main_image = np.zeros([*img_shape, 3])

    for n, current_sheet in enumerate(current_sheets):

        ### <font-shit>
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.75
        fontColor              = current_sheet.color # (200, 200, 200)
        lineType               = 1
        ### </font-shit>


        # concave hull method
        COMPUTE_CONCAVE_HULL = False
        if COMPUTE_CONCAVE_HULL:
            length = int(current_sheet.concave_hull_circumference())
        else:
            # convex hull method
            length = int(current_sheet.length)

        image = current_sheet.blit(img_shape = img_shape)
        image_height, image_width = image.shape[:2]


        #text_bottom_left = [np.max(current_sheet.xs), np.max(current_sheet.ys)]

        polygon = current_sheet.polygon

        xmean = int(np.mean(current_sheet.xs))
        text_x = xmean
        text_y = int(np.max(current_sheet.ys)) + 30
        main_image = current_sheet.blit(img_array = main_image)


        TRY_TO_MOVE_TEXT_CLOSER_TO_CLUSTER = True

        # <text>
        if current_sheet.id <= 6:

            if TRY_TO_MOVE_TEXT_CLOSER_TO_CLUSTER:

                while text_x > 0.90*image_width:
                    text_x -= 5

                point = Point((text_x, text_y))
                distance = polygon.distance(point)
                while distance < 40:
                    text_y += 5
                    point = Point((text_x, text_y))
                    distance = polygon.distance(point)

            text_bottom_left = (text_x, text_y)

            cv2.putText(
                main_image,
                f"length: {length}",
                text_bottom_left,
                font, 
                fontScale,
                fontColor,
                lineType,
            )
        # </text>

    cv2.imwrite(output_path, main_image)
    print(f"Wrote {output_path}")
    return current_sheets


if __name__ == '__main__':

    colors = get_unique_colors(n = 1000)

    turbulence_data_dir = os.path.join(os.getenv('DATA'), 'turbulence')
    #turbulence_data_dir = 'images_gray'
    #gda_files = set(glob.glob(f"{turbulence_data_dir}/*.gda"))
    #gda_files = sorted(gda_files, key = get_number_from_gda_pathname)
    #args_list = list(enumerate(gda_files[120:]))
    #args_list = list(enumerate(gda_files[270:370:3]))
    #args_list = list(enumerate(gda_files[120:]))
    gda_files = sorted(set(glob.glob(f"{turbulence_data_dir}/*.png")))
    args_list = list(enumerate(gda_files))

    OUTPUT_DIR = os.path.realpath('turbulence-output')
    os.makedirs(OUTPUT_DIR, exist_ok = True)

    parallel = False
    if parallel:
        with mp.Pool(mp.cpu_count()) as pool:
            pool.starmap(solve_turbulence, args_list)

    else:
        previous_current_sheets = []
        for image_num, image_path in args_list:
            previous_current_sheets = solve_turbulence(image_num, image_path, previous_current_sheets)

    #CMD = f"ffmpeg -f image2 -r 10 -i {OUTPUT_DIR}/%05d.png -vcodec mpeg4 -y -qscale:v 1 turbulence.mp4"
    #os.popen(CMD).read()

