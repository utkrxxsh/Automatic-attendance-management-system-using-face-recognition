import cv2
import numpy as np
import face_recognition

imgMatTrain = face_recognition.load_image_file('Res/matthewmcconaugheytrain.png')
imgMatTrain = cv2.cvtColor(imgMatTrain, cv2.COLOR_BGR2RGB)
imgMatTest = face_recognition.load_image_file('Res/matthewmcconaugheytest.jpg')
imgMatTest = cv2.cvtColor(imgMatTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgMatTrain)[0]
encodeMat = face_recognition.face_encodings(imgMatTrain)[0]
cv2.rectangle(imgMatTrain,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgMatTest)[0]
encodeMatTest = face_recognition.face_encodings(imgMatTest)[0]
cv2.rectangle(imgMatTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results=face_recognition.compare_faces([encodeMat],encodeMatTest)
faceDis=face_recognition.face_distance([encodeMat],encodeMatTest)
print(faceDis)
print(results)
cv2.putText(imgMatTest,str(results),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow("Matthew McConaughey Train", imgMatTrain)
cv2.imshow("Matthew McConaughey Test", imgMatTest)
cv2.waitKey(0)
