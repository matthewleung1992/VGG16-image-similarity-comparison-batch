# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:14:55 2023

@author: matt
"""
from multiprocessing import set_start_method
import multiprocessing
import threading
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
import logging
from os import getpid
import warnings
import time
import math

#Logger per worker
def setup_process_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')

#Feature extraction
def extract(img, model):
    img = load_img(img)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    return feature
    # return feature / np.linalg.norm(feature)

#For using thread
def thread_function(in_queue, out_queue):
    setup_process_logger()

    base_model = VGG16(weights='imagenet')

    layernames = [ #change to add/remove layers of interest
        'block1_conv1',
        'block1_pool',
        'block2_pool',
        'block3_conv3',
        'block3_pool',
        'block4_conv1',
        'block4_pool',
        'block5_pool',
        'flatten',
        'fc1',
        'fc2'
    ]

    layer_outputs = [base_model.get_layer(layer_name).output for layer_name in layernames]
    model = Model(inputs=base_model.input, outputs=layer_outputs)

    while True:
        in_arr = in_queue.get()
        if in_arr is None:
            break

        try:
            img1, img2 = in_arr
            # Set up dict
            feature_values1 = {}
            feature_values2 = {}
            img_feature1 = extract(img1, model)
            img_feature2 = extract(img2, model)
            for i in range(len(layernames)):
                feature_values1[layernames[i]] = img_feature1[i][0] / np.linalg.norm(img_feature1[i][0])
                feature_values2[layernames[i]] = img_feature2[i][0] / np.linalg.norm(img_feature2[i][0])

            # Calculate eu distances
            euclidean_distances = {}
            distance_dict = {}

            for layer in feature_values1:
                distance_dict[layer] = np.linalg.norm(feature_values1[layer] - feature_values2[layer])

            img1name = os.path.basename(img1)
            img2name = os.path.basename(img2)
            euclidean_distances[(img1name, img2name)] = distance_dict

            logging.info("Process {} processed {} and {}.".format(getpid(), img1name, img2name))

            out_queue.put(euclidean_distances)

        except Exception as e:
            logging.error("An error occurred in thread_function: %s", str(e))


if __name__ == '__main__':

    # Filter all warning messages
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel('FATAL')
    # Manual fix for multiprocessing deadlock bug in old Python versions
    # set_start_method("spawn")

    # IMGPATH MAKER start
    directory = os.getcwd()  # Specify the directory path HERE

    imgpoolpaths = []

    for root, dirs, files in os.walk(directory):
        jpg_files = [file for file in files if file.lower().endswith(".jpg")]
        for i, file in enumerate(jpg_files):
            if i >= 3:
                break
            f = os.path.join(root, file)
          
            if os.path.isfile(f):
                imgpoolpaths.append(f)

    input_csv_file = "results.csv"

    start_time = time.time()

    # Check if the CSV file exists and load data from the file
    if os.path.isfile(input_csv_file):
        df = pd.read_csv(input_csv_file)

        # Set the first two columns as a tuple index
        df.set_index(['Unnamed: 0', 'Unnamed: 1'], inplace=True)

        # Convert the DataFrame to a dictionary
        finalresults = df.to_dict(orient='index')

    else:
        proceed = input(f"Warning: The CSV file '{input_csv_file}' does not exist. Proceed? Y/N")

        if proceed != "Y":
            raise SystemExit

        finalresults = {}

    image_names = [os.path.basename(file) for file in imgpoolpaths]

    num_images = len(imgpoolpaths)
    print(f"Number of images: {num_images}")

    # IMGPATH MAKER end

    # MISSION LIST start

    missionlist = []

    # Iterate and find duplicates
    for i in range(num_images):

        img1 = image_names[i]

        for j in range(i + 1):

            img2 = image_names[j]

            if os.path.dirname(imgpoolpaths[i]) == os.path.dirname(imgpoolpaths[j]):    # No need to compare within dir
                continue

            # Check if the tuple already exists in finalresults
            if (img1, img2) in finalresults or (img2, img1) in finalresults:
                continue

            missionlist.append((imgpoolpaths[i], imgpoolpaths[j]))

    print(f"Mission: {len(missionlist)}")

    # MISSION LIST end

    # New code block start
    num_workers = 15
    write_threshold = 50
    in_queue_list = [multiprocessing.Queue(maxsize=128) for i in range(num_workers)]
    out_queue = multiprocessing.Queue()

    p_list = [multiprocessing.Process(target=thread_function, args=(in_queue_list[i], out_queue)) for i in range(num_workers)]
    [p.start() for p in p_list]

    def input_fn():
        for i, mission in enumerate(missionlist):
            in_queue_list[i % num_workers].put(mission)

        [in_queue_list[i].put(None) for i in range(num_workers)]

    input_thread = threading.Thread(target=input_fn)
    input_thread.start()

    time0 = time.time()

    for _ in range(len(missionlist)):
        result = out_queue.get()
        # Create a DataFrame from the calculated distances
        df = pd.DataFrame(result).T

        # Fill NaN values with an empty dictionary
        df.fillna({}, inplace=True)

        if os.path.isfile(input_csv_file):
            df.to_csv(input_csv_file, mode='a', header=False)
        else:
            df.to_csv(input_csv_file)

    input_thread.join()
    [p.join() for p in p_list]

    # Calculate the elapsed time
    elapsed_time = time.time() - time0
    print(f"The process took {elapsed_time} seconds.")
