import matplotlib
matplotlib.use("TkAgg") #Makes imshow work on mac
from matplotlib import pyplot as plt
from Model import Model, draw_boxes
import cozmo
import queue
import time
import threading
import numpy as np
import sys, os

def detectImages():
    print('Detect Images started')
    while(True):
        if(not imageQueue.empty()):
            try:
                plt.clf() # We need to clear the plot so that we are not plotting every image each iteration. (If we don't we will get increasing delay)

                preds, img_resized = model.object_detect(imageQueue.get())
                draw_boxes(preds, img_resized, (model.size, model.size))

                img_with_box = np.array(img_resized) 
                img_with_box = img_with_box[:, :, ::-1].copy() 

                plt.imshow(img_with_box)
                plt.pause(0.001) # imshow needs time to plot the image. Need this to display the image

            except queue.Empty:
                pass
    

def handle_image(evt, obj=None, tap_count=None,  **kwargs):
    try:
        if(imageQueue.empty()):
            imageQueue.put_nowait(evt.image)
    except queue.Full:
        pass

def configure_camera(robot, exposure_amount, gain_amount):
    robot.camera.color_image_enabled = True
    # Lerp exposure between min and max times
    min_exposure = robot.camera.config.min_exposure_time_ms
    max_exposure = robot.camera.config.max_exposure_time_ms
    exposure_time = (1-exposure_amount)*min_exposure + exposure_amount*max_exposure
    # Lerp gain
    min_gain = robot.camera.config.min_gain
    max_gain = robot.camera.config.max_gain
    actual_gain = (1-gain_amount)*min_gain + gain_amount*max_gain
    robot.camera.set_manual_exposure(exposure_time, actual_gain)
    

def cozmo_program(robot: cozmo.robot.Robot):
    robot.set_lift_height(1.0)
    exposure_amount = 0.3 # Range: [0,1]
    gain_amount = 0.9 # Range: [0,1]
    configure_camera(robot, exposure_amount, gain_amount)
    robot.add_event_handler(cozmo.camera.EvtNewRawCameraImage, handle_image)
    print("Added event handler")
    while True:
        time.sleep(0.1)


model = Model(path='../cs481-senior-design/f18/data/coco2014', jpegs='../cs481-senior-design/f18/train2014', bb_csv='../cs481-senior-design/f18/data/coco2014/tmp/bb.csv')
imageQueue = queue.Queue(maxsize=1)

threading.Thread(target=detectImages).start()

cozmo.run_program(cozmo_program, use_viewer=True)
