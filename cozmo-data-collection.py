import matplotlib
matplotlib.use("TkAgg") #Makes imshow work on mac
from matplotlib import pyplot as plt
from Model import Model, draw_boxes
import cozmo
import queue
import time
import threading
import numpy as np
import sys
import os
import pandas as pd
import base64
import CS481Dataset

def detectImages():
    global data
    imgCount = 0
    print('Detect Images started')
    while(True):
        if(not imageQueue.empty()):
            try:

                pilImg = imageQueue.get()
                box, imgResized = model.object_detect(pilImg.copy())
                cleanImg = imgResized.copy()
                draw_boxes(box, imgResized, (model.size, model.size))

                imgWithBox = np.array(imgResized) 
                imgWithBox = imgWithBox[:, :, ::-1].copy()
                box = box[0]
                formattedBox = (box[1], box[0], box[3], box[2]) # coordinates need to be corrected for crop
                imgBox = cleanImg.crop(formattedBox)
                plt.imshow(imgBox)
                plt.pause(0.001) # imshow needs time to plot the image. Need this to display the image

                print('Do you want to save this image? (y for yes, press enter for next image):')
                response = sys.stdin.readline()
                if response.strip() == 'y':
                    CS481Dataset.encodeAndSaveImage(data, pilImage, label, formattedBox)
                    imgCount+=1
                    print(label + '\n')
                    print('img count: ' + str(imgCount) + '\n')
                    data.to_csv(dataset, index=False)
                    print('saved' + '\n')
                
            except queue.Empty:
                pass
    

def handle_image(evt, obj=None, tap_count=None,  **kwargs):
    try:
        if(imageQueue.empty()):
            imageQueue.put_nowait(evt.image)
    except queue.Full:
        pass

def cozmo_program(robot: cozmo.robot.Robot):
    robot.camera.color_image_enabled = True
    robot.add_event_handler(cozmo.camera.EvtNewRawCameraImage, handle_image)
    robot.set_lift_height(1.0)
    print("Added event handler")
    while True:
        time.sleep(0.1)


model = Model(path='../cs481-senior-design/f18/data/coco2014', jpegs='../cs481-senior-design/f18/train2014', bb_csv='../cs481-senior-design/f18/data/coco2014/tmp/bb.csv')
imageQueue = queue.Queue(maxsize=1)

dataset = 'dataset.csv'

data = pd.read_csv(dataset)

print(data)

print('Enter Label:')
label = sys.stdin.readline().strip()

threading.Thread(target=detectImages).start()

cozmo.run_program(cozmo_program, use_viewer=True)
