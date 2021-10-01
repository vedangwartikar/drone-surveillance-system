########################### Importing the required libraries ###########################

import os
import numpy as np

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import matplotlib.pyplot as plt
import cv2
import time

import tensorflow as tf
from tensorflow.python.keras import backend as K

from google_drive_downloader import GoogleDriveDownloader as gdd

########################### Function for predicting for each frame of the video ###########################

def video_test(data):
    
    # Downloading online video to local file system
    
    if 'drive.google.com' in data:
        drive_id = data.replace('https://drive.google.com/file/d/','')
        gdd.download_file_from_google_drive(file_id=drive_id,
                                        dest_path='data/video1.mp4',
                                        overwrite=True)
    elif 'firebasestorage.googleapis.com' in data:
        import urllib.request
        urllib.request.urlretrieve(data, "data/video1.mp4")
    
    # Setting up tensorflow
    
    sess = tf.compat.v1.Session()
    K.set_session(sess)
    
    def get_session():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(config=config)
    
    tf.compat.v1.keras.backend.set_session(get_session())

    model_path = 'models/resnet50_csv_12_inference.h5'
    
    # Load retinanet model
    
    model = models.load_model(model_path, backbone_name='resnet50')
    
    # Load label to names mapping for visualization purposes
    
    labels_to_names = {0: 'Biker', 1: 'Car', 2: 'Bus', 3: 'Cart', 4: 'Skater', 5: 'Pedestrian'}
    
    print(data)
    sdd_images = os.listdir('data/video_frames/')
    prediction_values = []
    
    # Adding bounding boxes onto image usng the model
    
    def run_detection_image(filepath):
        image = read_image_bgr(filepath)
    
        # Copy to draw on image
        
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    
        # Preprocess image for network
        
        image = preprocess_image(image)
        image, scale = resize_image(image)
    
        # Checking process image time
        
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("Processing time: ", time.time() - start)
    
        
        boxes /= scale
        total_people = 0
        accuracy_list = []
        box_list = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.5:
                break
    
            color = label_color(label)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            b = box.astype(int)
            if caption.split()[0] in ['Pedestrian', 'Biker', 'Skater']:
                draw_box(draw, b, color=color)
                box_list.append(b)
                total_people += 1
                accuracy_list.append(float(caption.split()[1]))
                draw_caption(draw, b, caption)
        
        # Checking if 2 or more people are close to each other
        
        crowd = 0
        for i in range(len(box_list)):
            for j in range(i+1, len(box_list)):
                if box_list[i][0] < box_list[j][2] and box_list[j][0] < box_list[i][2] and box_list[i][1] < box_list[j][3] and box_list[j][1] < box_list[i][3]:
                    crowd += 1
        
        is_crowd = True if crowd else False
        print('Crowds: ', crowd)
        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()
        
        # Saving the predicted frames with bounding boxes
        
        file, ext = os.path.splitext(filepath)
        image_name = file.split('/')[-1] + ext
        output_path = os.path.join('data/predicted_frames/', image_name)
        
        draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path, draw_conv)
        print('Accuracy List : ', accuracy_list)
        print('Total people detected : ', total_people)
        
        return [total_people, is_crowd]
    
    base_path = 'data/video_frames/'
    
    # Capturing the video using opencv
    
    vidcap = cv2.VideoCapture('data/video1.mp4') 
    def getFrame(sec): 
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
        hasFrames,image = vidcap.read() 
        if hasFrames: 
            cv2.imwrite(base_path + "frame_"+str(sec)+"_sec.jpg", image)
        return hasFrames 
    sec = 0 
    frameRate = 1
    success = getFrame(sec) 
    while success: 
        sec = sec + frameRate 
        sec = round(sec, 2) 
        success = getFrame(sec) 
        
    print("Done!")
    
    #Predict for each frame extracted from the video
    
    time.sleep(2)
    
    prediction_values = []
    for image in sdd_images:
        print(image)
        if '.jpg' in image or '.png' in image:
            ret = run_detection_image(os.path.join('data/video_frames/',image))
            prediction_values.append(ret)
    
    print(prediction_values)
    
    # Prediction in the form [[no_of_people, is_crowd(Yes/No)], [no_of_people, is_crowd(Yes/No)].........]
    
    return prediction_values
        
########################### send visualization image linkto UI ###########################    

def send_visualization():
    
    #return 'https://drive.google.com/file/d/1naZlruKjl659iQjL8GgJ3srdGztxHNjO'

    return 'https://firebasestorage.googleapis.com/v0/b/drone-surveillance-fbe17.appspot.com/o/9cg8svn74dp?alt=media&token=34678601-8908-4b8a-a5b6-e9f31d8032ba'

def send_accuracy():
    
    first_line = open("accuracy.txt").readline().rstrip()
        
    return float(first_line[:6]) * 100