#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("C:\\Users\\91701\\Downloads\\archive (6)\\yolov3-obj_final.weights", "C:\\Users\\91701\\Downloads\\archive (6)\\yolov3_pb.cfg")
classes = []
with open("C:\\Users\\91701\\Downloads\\archive (6)\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image
img = cv2.imread("C:\\Users\\91701\\Downloads\\archive (6)\\Force-Traveller-Ambulance-1000x563.jpg")

# Check if image is loaded successfully
if img is None:
    print("Error: Unable to read the image.")
else:
    # Resize image
    img = cv2.resize(img, None, fx=0.4, fy=0.4)

    # Proceed with object detection
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 2:  # Assuming emergency vehicles class ID is 2
                # Object detected is an emergency vehicle
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw boxes on the image
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y + 30), font, 3, (0, 255, 0), 3)

    # Display image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[1]:


import cv2
import numpy as np


# In[2]:


net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# In[3]:


import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pylab as pl
from PIL import Image


# In[19]:


img0 = "C:\\Users\\91701\\Downloads\\archive (6)\\__results___2_0.png"
_ = plt.figure(figsize = (15,15))
_ = plt.axis('off')
_ = plt.imshow(mpimg.imread(img0))


# In[21]:


directory = "C:\\Users\\91701\\Downloads\\archive (6)\\dataset\\obj"

imagepath=[]
imagefile=[]
boxset=[]
boxfile=[]

for im in os.listdir(directory):
    if im[-4:]=='.jpg':
        path=os.path.join(directory,im)
        imagepath+=[path]
        imagefile+=[im]
        
for im in imagefile:
    if im[-4:]=='.jpg':
        bx=im[0:-4]+'.txt'
        path=os.path.join(directory,bx)
        if os.path.isfile(path):
            bxdata=np.loadtxt(path)
        boxset+=[bxdata]
        boxfile+=[bx] 


# In[22]:


print(imagefile[0:5])
print(boxfile[0:5])


# In[23]:


print(len(boxset))
print(len(imagepath))
795


# In[24]:


num0=0
for i in range(692):
    if imagepath[i]==img0:
        num0=i
        print(i)


# In[25]:


# for person on bike
weights0_path = "C:\\Users\\91701\\Downloads\\archive (6)\\yolov3-obj_final.weights"
configuration0_path = "C:\\Users\\91701\\Downloads\\archive (6)\\yolov3_pb.cfg"

probability_minimum = 0.5
threshold = 0.3


# In[27]:


network0 = cv2.dnn.readNetFromDarknet(configuration0_path, weights0_path)
layers_names0_all = network0.getLayerNames()
layers_names0_output = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
labels0 = open("C:\\Users\\91701\\Downloads\\archive (6)\\coco.names").read().strip().split('\n')
print(labels0)


# In[28]:


weights1_path = "C:\\Users\\91701\\Downloads\\archive (6)\\yolov3-obj_final.weights"
configuration1_path = "C:\\Users\\91701\\Downloads\\archive (6)\\yolov3_pb.cfg"


# In[29]:


network1 = cv2.dnn.readNetFromDarknet(configuration1_path, weights1_path)
layers_names1_all = network1.getLayerNames()
layers_names1_output = [layers_names1_all[i[0]-1] for i in network1.getUnconnectedOutLayers()]
labels1 = open('../input/helmet-detection-yolov3/helmet.names').read().strip().split('\n')
print(labels1)


# In[ ]:




