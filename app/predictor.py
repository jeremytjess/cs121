from __future__ import division, print_function
import sys
import os
import glob
import re
from pathlib import Path
from PIL import Image

# Import fast.ai Library
from fastai import *
from fastai.vision import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


def load_model(model_path):
    """
        load_model will return the model loaded
        with the weights calculated from stage 2
    """
    path = Path(model_path)
    classes = ['Angry', 'Surprised','Happy', 'Sad']
    data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms()).normalize(imagenet_stats)
    learn = create_cnn(data2, models.resnet50)
    learn.load('stage-2-rn50')
    return learn


def model_predict(img_path, model_path):
    """
       model_predict will return the preprocessed image
    """
    learn = load_model(model_path)
    #open image and convert to grayscale 
    img = open_image(img_path, convert_mode='L')
    # resize to 48 x 48
    img = img.resize((0,48,48))
    pred_class,pred_idx,outputs = learn.predict(img)
    return pred_class

