
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset , DataLoader 
import albumentations as a 
import cv2 
import matplotlib.pyplot as plt 
import timm 
import numpy as np
from torchvision import transforms
from glob import glob
import gradio as gr
import os 

#from example import FeatureExtractor
from Covid_pne import FeatureExtractor
from Covid_pne  import dataset
from Covid_pne  import ModelOutputs
from Covid_pne  import GradCam
from Covid_pne  import Prediction



#pa= [r'C:\Users\ACER\Desktop\Health tech\images\Viral Pneumonia-7.png']
#obj = Prediction(pa)
#print(pa)
#pred = obj.show_prediction() 
#print(pred)



