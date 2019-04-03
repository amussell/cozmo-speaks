import numpy as np
import pandas as pd
import cv2
from fastai.conv_learner import *
from fastai.dataset import *
from PIL import ImageDraw, ImageFont, Image

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax
def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)
def draw_im(im, ann):
    ax = show_img(im, figsize=(16,8))
    for b,c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)
#def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1],a[2]])
#def hw_bb(bb): return np.array([bb[1], bb[0], bb[3]+bb[1]-1, bb[2]+bb[0]-1])
def hw_bb(bb):
  ret = np.array([bb[1], bb[0], bb[3], bb[2]])
  return ret

def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    #ratio = 0.01
    #box = box.reshape(2, 2) * ratio
    box = box.reshape(2, 2)
    tmp = box[0][0]
    ret = box.reshape(-1)
    ret = hw_bb(ret)
    return ret

def draw_boxes(box, img, detection_size):
    draw = ImageDraw.Draw(img)
    color = tuple(np.random.randint(0, 256, 3))
    #box2 = bb_hw(box[0])
    box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
    draw.rectangle(box, outline=color)

    #for cls, bboxs in boxes.items():
    #    color = tuple(np.random.randint(0, 256, 3))
    #    for box, score in bboxs:
    #        box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
    #        draw.rectangle(box, outline=color)

class Model:

    def __init__(self, path: str, jpegs: str, bb_csv: str):
        self.f_model = resnet34
        self.size = 224
        self.bs = 64
        augs = [RandomFlip(tfm_y=TfmType.COORD),
                RandomRotate(30, tfm_y=TfmType.COORD),
                RandomLighting(0.1,0.1, tfm_y=TfmType.COORD)]

        tfms = tfms_from_model(self.f_model, self.size, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=augs)
        md = ImageClassifierData.from_csv(str(path), str(jpegs), str(bb_csv), tfms=tfms, continuous=True, bs=4)
        head_reg4 = nn.Sequential(Flatten(), nn.Linear(25088,4))
        self.learn = ConvLearner.pretrained(self.f_model, md, custom_head=head_reg4)
        self.learn.opt_fn = optim.Adam
        self.learn.crit = nn.L1Loss()

        self.learn.load('reg7')
        
    def object_detect(self, img: PIL.Image):
        tempFile = 'outfile.jpg'
        img_resized = img.resize(size=(self.size, self.size))
        scipy.misc.imsave(tempFile, img_resized)

        open_cv_image = np.array(img_resized) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        trn_tfms, val_tfms = tfms_from_model(self.f_model, self.size)
        
        im = val_tfms(open_image(tempFile))
        preds = to_np(self.learn.predict_array(im[None]))

        return preds, img_resized