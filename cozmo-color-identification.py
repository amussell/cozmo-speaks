import matplotlib
matplotlib.use("TkAgg") #Makes imshow work on mac
from matplotlib import pyplot as plt
from Model import Model, draw_boxes
import collections
import LanguageModel
import cozmo
import pickle
import sys
import queue
import time
import threading
import numpy as np

def detectImages():
    global predicting
    print('Detect Images started')
    while(True):
        if(not imageQueue.empty()):
            try:
                plt.clf() # We need to clear the plot so that we are not plotting every image each iteration. (If we don't we will get increasing delay)


                pilImg = imageQueue.get()
                box, imgResized = model.object_detect(pilImg.copy())
                cleanImg = imgResized.copy();
                draw_boxes(box, imgResized, (model.size, model.size))

                imgWithBox = np.array(imgResized) 
                #imgWithBox = imgWithBox[:, :, ::-1].copy()
                box = box[0]
                formattedBox = (box[1], box[0], box[3], box[2]) # coordinates need to be corrected for crop
                croppedImg = cleanImg.crop(formattedBox)
                croppedImg = np.array(croppedImg) 
                croppedImg = croppedImg[:, :, ::-1].copy()

                if predicting == False:
                    imageBuffer.append(croppedImg)

                plt.imshow(imgWithBox)
                plt.pause(0.001) # imshow needs time to plot the image. Need this to display the image

            except queue.Empty:
                pass

def predict():
    global speak
    global prediction
    preds = []
    for i in range(len(imageBuffer)):
        currentImg = imageBuffer.pop()
        preds.append(predModel.predictImageWord(currentImg))
    preds = [item[0] for item in preds] #remove probability number in tuple
    prediction = collections.Counter(preds).most_common(1)[0][0]
    speak = True
    
def key_listener():
    global predicting
    while True:
        print('Predict?')
        response = sys.stdin.readline()
        if response.strip() == 'y':    
            predicting = True
            predict()
            predicting = False

def handle_image(evt, obj=None, tap_count=None,  **kwargs):
    try:
        if(imageQueue.empty()):
            imageQueue.put_nowait(evt.image)
    except queue.Full:
        pass

def cozmo_program(robot: cozmo.robot.Robot):
    robot.set_lift_height(1.0).wait_for_completed()
    robot.camera.color_image_enabled = True
    robot.add_event_handler(cozmo.camera.EvtNewRawCameraImage, handle_image)
    print("Added event handler")
    #robot.say_text("purple").wait_for_completed()
    while True:
        global speak
        if speak == True:
            robot.say_text(prediction).wait_for_completed()
            speak = False
        time.sleep(0.1)


model = Model(path='../cs481-senior-design/f18/data/coco2014', jpegs='../cs481-senior-design/f18/train2014', bb_csv='../cs481-senior-design/f18/data/coco2014/tmp/bb.csv')

with open('../cs481-senior-design/s19/language-model.pickle', 'rb') as handle:
    predModel = pickle.load(handle) #trying to load a LanguageModel type

imageQueue = queue.Queue(maxsize=1)
imageBuffer = collections.deque(maxlen=8)
speak = False
predicting = False
prediction = ''

threading.Thread(target=detectImages).start()
threading.Thread(target=key_listener).start()

cozmo.run_program(cozmo_program, use_viewer=True)
