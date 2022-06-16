from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import random
import sys
import cv2
import pydicom as dicom
import os
from imageio.v2 import imread
import pandas as pd
import re

sys.path.append('./funcs/')
sys.path.append('./nets/')
import subprocess
import time
from shutil import rmtree
from optparse import OptionParser
from echoanalysis_tools import output_imgdict

import vgg as network

# # Hyperparams
parser = OptionParser()
parser.add_option("-d", "--dicomdir", dest="dicomdir", help="dicomdir")
parser.add_option("-g", "--gpu", dest="gpu", default="0", help="cuda device to use")
parser.add_option("-m", "--model", dest="model")
params, args = parser.parse_args()
dicomdir = params.dicomdir
model = params.model

#import vgg as network

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu


def dicom2jpg(path, dicomfile, out_dir):
    '''
    uses python GDCM to convert a compressed dicom into jpg images
    :param path: path to dicom file
    :param dicomfile: name of dicom file
    :param out_dir: output directory to put jpgs
    '''

    ds = dicom.dcmread(os.path.join(path, dicomfile))
    if dicomfile[-4:] == '.dcm':
        dicomfile = dicomfile.replace('.dcm', '.jpg')
    else:
        dicomfile = dicomfile + '.jpg'

    try:
        pixel_array_numpy = ds.pixel_array
        counter = 0
        for img_array in pixel_array_numpy:
            cv2.imwrite(os.path.join(out_dir, dicomfile.replace('.jpg', '-{}.jpg'.format(counter))), img_array)
            counter += 1
    except AttributeError:
        print(dicomfile + " failed: no Pixel Data")
        pass
    return


def extract_jpg_single_dicom(dicom_directory, out_directory, filename):
    '''
    Functional to convert dicom to jpg
    :param: dicom_directory: directory of dicoms
    :param out_directory: directory to place jpgs (expect this to be dicom_directory/image/)
    :param filename: name of dicom file
    '''

    filepath = os.path.join(dicom_directory, filename)
    print(filepath, "trying")
    time.sleep(2)
    ds = dicom.dcmread(filepath, force=True)
    framedict = output_imgdict(ds)

    # some quick error handling
    if framedict == "Only Single Frame":
        print("Single Frame DICOM skipped: {}".format(filename))
        return
    if framedict == "General Failure":
        print("DICOM skipped, likely has an attribute missing: {}".format(filename))
        return

    y = len(framedict.keys()) - 1
    try:
        ds = dicom.dcmread(filepath)
        framedict = output_imgdict(ds)
        y = len(framedict.keys()) - 1
        if y > 10:
            m = random.sample(range(0, y), 10)
            for n in m:
                targetimage = framedict[n][:]
                outfile = os.path.join(out_directory, filename) + '-{}.jpg'.format(n)
                cv2.imwrite(outfile, cv2.resize(targetimage, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            print("Too few frames: {}".format(filename))

    except (IOError, EOFError, KeyError) as e:
        print(out_directory + "\t" + filename + "\t" +
              "error", e)
    return None


def extract_imgs_from_dicoms(dicom_directory, out_directory, filenames=None):
    """
    Extracts jpg images from DCM files in the given directory

    @param dicom_directory: folder with DCM files of echos
    @param out_directory: destination folder to where converted jpg files are placed
    @param target: destination folder to where converted jpg files are placed
    """
    if filenames is None:
        filenames = os.listdir(dicom_directory)

    for filename in filenames[:]:
        if filename == 'image': # skip if its the temp directory we've made called "image/"
            pass
        else:
            extract_jpg_single_dicom(dicom_directory, out_directory, filename)
    return 1


def classify(directory, feature_dim, label_dim, model_name):
    """
    Classifies jpg echo images in given directory

    @param directory: folder with jpg echo images for classification
    """
    imagedict = {}
    predictions = {}
    for filename in os.listdir(directory):
        if "jpg" in filename:
            image = imread(directory + filename).astype('uint8').flatten()
            imagedict[filename] = [image.reshape((224, 224, 1))]

    tf.reset_default_graph()
    sess = tf.Session()
    model = network.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_name)

    for filename in imagedict:
        predictions[filename] = np.around(model.probabilities(sess, imagedict[filename]), decimals=3)

    return predictions


def main(dicomdir = "/Users/jameswilkinson/Documents/FeinbergData/2022-05-22/dicoms/",
         batch_size=100,
         model_name="view_23_e5_class_11-Mar-2018",
         model_path='./echo_deeplearning/models/'):

    '''

    :param dicomdir: directory containing dicoms
    :param batch_size: number of dicoms to process in one go. If less than the number of dicoms in the
                    dicomdir, then the output results file will be writen multiple times, once after each
                    batch runs.
    :param model_name: name of the model. Try downloading the model prefix'd "view_23_e5_class_11-Mar-2018"
                from https://www.dropbox.com/sh/0tkcf7e0ljgs0b8/AACBnNiXZ7PetYeCcvb-Z9MSa?dl=0. You will need
                three files, one with each of the extensions ['.data-00000-of-00001', '.index', '.meta'].
                All three files should be inside the directory at model_path
    :param model_path: path to the model
    :return: None, but writes results to a csv in ./results/
    '''

    model_pathname = os.path.join(model_path, model_name)

    infile = open("viewclasses_" + model_name + ".txt")
    infile = infile.readlines()
    views = [i.rstrip() for i in infile]

    feature_dim = 1
    label_dim = len(views)

    out = pd.DataFrame(index=None, columns=['study', 'model'] + ["prob_{}".format(v) for v in views])

    x = time.time()
    temp_image_directory = os.path.join(dicomdir, 'image/')
    if os.path.exists(temp_image_directory):
        rmtree(temp_image_directory)
    if not os.path.exists(temp_image_directory):
        os.makedirs(temp_image_directory)

    RemainingDicoms = os.listdir(dicomdir)
    while len(RemainingDicoms) > 0: # we want to process the dicoms in batches versus all at once
        dicoms = RemainingDicoms[:min(batch_size, len(RemainingDicoms))] # filenames of dicoms in current batch

        if len(dicoms) < len(RemainingDicoms): # if theres any left, trim down the stack
            RemainingDicoms = RemainingDicoms[batch_size:] # remove these from the stack
        else:
            RemainingDicoms = [] # if we've used all the dicoms, set list to an empty one

        # 1) extract jpg images from dicoms into temp_image_directory
        extract_imgs_from_dicoms(dicomdir, temp_image_directory, filenames=dicoms)

        # 2) generate predictions
        predictions = classify(temp_image_directory, feature_dim, label_dim, model_pathname)

        # 3) write to the results, and save as csv
        predictprobdict = {}
        for imagename in predictions.keys():
            prefix = re.split('-[0-9]+.jpg', imagename)[0] # name of dicom file (not incl. the frame number)
            if prefix not in predictprobdict:
                predictprobdict[prefix] = []
            predictprobdict[prefix].append(predictions[imagename][0])
        for prefix in predictprobdict.keys():
            predictprobmean = np.mean(predictprobdict[prefix], axis=0)
            predictprobdict[prefix] = predictprobmean # replace with mean of all predictions
            fulldata_list = [prefix, model_name] + list(predictprobmean)
            out.loc[len(out) + 1] = fulldata_list

        _dicompathtemp = os.path.normpath(dicomdir)
        output_file_name = '_'.join(_dicompathtemp.split(os.sep)[-2:])
        print("Predictions for {} with {} \n {}".format(dicomdir, model_name, out))
        out.rename(output_file_name, inplace=True)
        out.to_csv(index=False)

        # 4) empty the tmp directory of jpgs
        for jpg in os.listdir(temp_image_directory):
            filepath = os.path.join(temp_image_directory, jpg)
            rmtree(filepath)

    y = time.time()
    print("time:  " + str(y - x) + " seconds for " + str(len(predictprobdict.keys())) + " videos")
    rmtree(temp_image_directory)
    return None


if __name__ == '__main__':
    # dicomdir needs to be a directory containing dicoms. There could be multiple of these, just loop through if needed
    # batch_size limits the number of dicoms that are processed in one go. This can help out if a directory has hundreds
    #    of dicoms, which could take days to process. batch_size means the code will write and save a user-accessible
    #    output on-the-fly, processing in smaller batches as it goes.
    main(dicomdir="/Users/jameswilkinson/Documents/FeinbergData/2022-05-22/dicoms/",
        batch_size=10)
