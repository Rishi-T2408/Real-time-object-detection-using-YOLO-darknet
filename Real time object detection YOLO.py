import cv2,time
import numpy as np
import os
cap=cv2.VideoCapture(0)      #vedio object for assesing webcam

wht=320
confThreshold=0.5
nmsThreshold=0.3
classesFile='coco.names'
classNames=[]
with open(classesFile,"rt") as f:
    classNames = f.read().rstrip('\n').split('\n')

    modelConfurigation='yolov3.cfg'    #importing the yolo confurigation on code
    modelweights='yolov3.weights'

    #Now we are creating an network to work on real time object detection
    net= cv2.dnn.readNetFromDarknet(modelConfurigation,modelweights)
    #now we creating cv2 as backend for our source code
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    def findobjects(outputs,img):
        hT,wT,cT =img.shape
        bbox=[]
        classIds=[]
        confs=[]

        for output in outputs:
            for det in output:
                scores=det[5:]
                classId=np.argmax(scores)
                confidence=scores[classId]
                if confidence>confThreshold:
                    w,h=det[2]*wT,det[3]*hT
                    x,y=int((det[0]*wT)-w/2),int((det[0]*hT)-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        #print(len(bbox))
        indices= cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
        print(indices)
        for i in indices:
            i=i[0]
            box=bbox[i]
            x,y,w,h=box[0],box[1],box[2],box[3]
            cv2.rectangle(img , (int(x),int(y)) , (int(x+w),int(y+h)) , (1,100,0) , 2)
            rr=classNames[classIds[i]].upper()
            tt=int(confs[i]*100)
            cv2.putText(img,"{0}  {1}%".format(rr,tt), (x, y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (20, 200, 0), 3)





#we have to convert our image to blob which the network accepts
while True:
    success,img=cap.read()
    blob=cv2.dnn.blobFromImage(img,1/255,(wht,wht),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames=net.getLayerNames()
    outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)

    #print(net.getUnconnectedOutLayers())
    outputs=net.forward(outputNames)
   # print((outputs[0]).shape) #hmari pics johhh numpy array mai convert hooti hai uska shape hai means dimentions
   # print((outputs[1]).shape)
    #print((outputs[2]).shape)
    #print(outputs[0][0])

    findobjects(outputs,img)


    cv2.imshow('Image',img)
    cv2.waitKey(1)
