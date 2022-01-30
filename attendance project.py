#!/usr/bin/env python
# coding: utf-8

# In[47]:


import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'Downloads/ImagesForAttendance/'
images = []
classnames = []
mylist = os.listdir(path)


for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)    
print(mylist) 


# In[48]:


def findEncodings(images):
    encodelist = []
    
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        
        encodelist.append(encode)
    return encodelist

encodeListKnown = findEncodings(images)

#print(len(encodeListKnown))


def markAttendance(name):
    with open('Attendance/Attendance.csv','r+') as f:
        myDataList = f.readlines()
        namelist = []
        
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            
            


# In[49]:


vc = cv2.VideoCapture(0)

while True:
    
    ret,frame = vc.read()
    frameS = cv2.resize(frame,(0,0),None,.25,.25)
    frameS = cv2.cvtColor(frameS,cv2.COLOR_BGR2RGB)
    
    facesLocCurFrame = face_recognition.face_locations(frameS)
    encodeCurFrame = face_recognition.face_encodings(frameS,facesLocCurFrame)
    
    for faceloc,encodeface in zip(facesLocCurFrame,encodeCurFrame):
        
        matches = face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeface) 
        
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)
            
            y1,x2,y2,x1 = faceloc
            
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-30),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            markAttendance(name)
    
     
    
    cv2.imshow('webcam',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
vc.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    


# In[ ]:




