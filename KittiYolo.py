# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import argparse
import sys
import numpy as np
import os
import os.path
import time
#inputPath = '../MyDataset/InputPath/'
#outImagePath = '../MyDataset/OutputPath/'
# outTxtPath = '../MyDataset/OutTxtPath/'
#/home/hu/MyProject/MyDataset/Kitti/secquence/00/image_2/
#/home/hu/MyProject/MyDataset/Kitti/secquence/00/Dection/
inputPath = '../MyDataset/Kitti/secquence/10/image_2/'
outTxtPath = '../MyDataset/Kitti/secquence/10/Dection/'

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image
parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outTxt, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        outTxt.write(str(classIds[i]) + '\t'
                     + str(confidences[i]) + '\t'
                     + str(left) + '\t'
                     + str(top) + '\t'
                     + str(left + width)+ '\t'
                     + str(top + height) + '\n')
        #draw
        #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    outTxt.close()

# Process inputs
#winName = 'Deep learning object detection in OpenCV'
#cv.namedWindow(winName, cv.WINDOW_NORMAL)
i = 1
NumImage = len(os.listdir(inputPath)) - 1
print("===================================")
start = time.time()
for imageName in os.listdir(inputPath):
    frame = cv.imread(inputPath + imageName)
    if frame is None:
        print("Input image file ", imageName, " doesn't exist")
        continue

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    outTxt = open(outTxtPath + imageName[:-4] + ".txt", 'w+')
    # Remove the bounding boxes with low confidence
    postprocess(frame, outTxt, outs)
    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    #t, _ = net.getPerfProfile()
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    #outImage = imageName[:-4] + '_yolo_out.png'


    #cv.imwrite(outImagePath + outImage, frame.astype(np.uint8))
    #cv.imshow(winName, frame)

    timeUsed = round(time.time() - start)
    #print("耗时:", '%.2f' % (timeUsed),"s,已处理： ", i, "/", NumImage)
    if (i % 10 == 0):
        #hours = timeUsed // 3600
        #minutes = (timeUsed // 60) % 60
        minutes = timeUsed // 60
        seconds = timeUsed % 60
        timePerImage = timeUsed / i
        timeRest = round(timePerImage * (NumImage - i))
        print("-------------耗时:",minutes, "min", seconds, 's.')
        print("-------------已处理：",i,"/",NumImage,'帧.')
        print("-------------平均帧率：",'%.2f' %(1/timePerImage),"fps.")
        #hours = timeRest // 3600
        #minutes = (timeRest // 60) % 60
        minutes = timeRest // 60
        seconds = timeRest % 60
        print("-----------------------------------")
        #print("---预计剩余: ", hours, "h", minutes, "min", seconds, 's.')
        print("---   预计剩余: ", minutes,"min", seconds,'s   ---')
        #print("--------------预计剩余：",'%.2f' % (timeRest/60) + "min.")
        print("===================================")
        print("=============shishi=================")
    i = i + 1
    #cv.waitKey(0)
