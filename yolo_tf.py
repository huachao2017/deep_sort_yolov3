#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np

import tensorflow as tf
from yolo3.utils import letterbox_image
from model import yolov3
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.data_aug import letterbox_resize
import cv2


class YOLO_TF(object):
    def __init__(self,is_tiny = False):
        if is_tiny:
            self.model_path = './model_data/??' #TODO
            self.anchors_path = './model_data/yolo_tiny_anchors.txt'
        else:
            self.model_path = './model_data/yolov3.ckpt'
            self.anchors_path = './model_data/yolo_anchors.txt'

        self.classes_path = './model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.45
        self.class_names = read_class_names(os.path.expanduser(self.classes_path))
        self.num_class = len(self.class_names)
        self.anchors = parse_anchors(os.path.expanduser(self.anchors_path))

        self.model_image_size = (416, 416) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.load()

    def load(self):

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3  # 占用GPU30%的显存
        self._graph = tf.Graph()
        with self._graph.as_default():
            self.input_data = tf.placeholder(tf.float32, [1, self.model_image_size[1], self.model_image_size[0], 3], name='input_data')
            yolo_model = yolov3(self.num_class, self.anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = yolo_model.forward(self.input_data, False)
            pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

            pred_scores = pred_confs * pred_probs

            self.boxes, self.scores, self.classes = gpu_nms(pred_boxes, pred_scores, self.num_class, max_boxes=200, score_thresh=self.score,
                                            nms_thresh=self.iou)

            saver = tf.train.Saver()

            self.sess = tf.Session(config=config)
            saver.restore(self.sess, os.path.expanduser(self.model_path))


    def detect_image(self, frame):
        t1 = time.time()
        img, resize_ratio, dw, dh = letterbox_resize(frame, self.model_image_size[0], self.model_image_size[1])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.


        t2 = time.time()
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.input_data: img
            })
        t3 = time.time()
        out_boxes[:, [0, 2]] = (out_boxes[:, [0, 2]] - dw) / resize_ratio
        out_boxes[:, [1, 3]] = (out_boxes[:, [1, 3]] - dh) / resize_ratio
        return_boxs = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person' :
                continue
            box = out_boxes[i]
           # score = out_scores[i]  
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]-box[0])
            h = int(box[3]-box[1])
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0 
            return_boxs.append([x,y,w,h])

        t4 = time.time()
        print("yolo:before,detect,after= %.3f:%.3f,%.3f,%.3f" % (t4 - t1, t2 - t1, t3 - t2, t4 - t3))
        return return_boxs

    def close_session(self):
        self.sess.close()
