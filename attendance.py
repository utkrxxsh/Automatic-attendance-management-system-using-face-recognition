import cv2
import numpy
import numpy as np
import face_recognition
import os
from datetime import datetime

path = "Attn"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    root_ext = os.path.splitext(cl)
    classNames.append(root_ext[0])

print(classNames)


def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def markAtt(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        timeList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            timeList.append(entry[1])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


knownencodeList = findEncoding(images)
print('Encoding completed')

cap = cv2.VideoCapture(1)
while True:
    _, img = cap.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(img_small)
    encodingsCurrFrame = face_recognition.face_encodings(img_small, facesCurrFrame)

    for encodeFace, faceLoc in zip(encodingsCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(knownencodeList, encodeFace)
        faceDis = face_recognition.face_distance(knownencodeList, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            x1, y1, x2, y2 = x1 * 4, y1 * 4, x2 * 4, y2 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            markAtt(name)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
