import cv2
import numpy as np

net= cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg.txt')

classes=[]
with open('coco.names.txt','r') as f:
    classes= f.read().splitlines()
#print(classes)

#img = cv2.imread('1.jpg')
cap = cv2.VideoCapture("cars.mp4")
while(1):
    ret, img =cap.read()
    #img=np.flip(img,axis=1) 

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img,1/255,(416,416),(0,0,0),swapRB=True,crop=False)
    '''
    for b in blob:
        for n,output_pics in enumerate(b):
            cv2.imshow(str(n),output_pics)
    '''
    net.setInput(blob)
    output_layer_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layer_names)

    boxes,confidences,class_ids=[],[],[]

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if(confidence>0.5):
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    #print(len(boxes))
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    print(type(indexes),len(indexes))
    #indexes=indexes.flatten()

    font = cv2.FONT_HERSHEY_PLAIN
    colors= np.random.uniform(0, 255, size=(len(boxes),3))

    if(len(boxes)>0):
        indexes=indexes.flatten()
        for i in indexes:
            x,y,w,h=boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],4))
            color= colors[i]
            cv2.rectangle((img),(x,y),(x+w,y+h),color,2)
            cv2.putText((img),label+" "+str(float(confidence)*100)+str('%'),(x,y+20),font,2,(0,0,0),2)
            cv2.putText((img),str(len(boxes)),(550,50),font,2,(0,0,0),5)

        
    cv2.imshow('Detected Objects',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
