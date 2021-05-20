
# Automatic-attendance-management-system-using-face-recognition
Attendance management system using OpenCV and python . The attendance system uses webcam to record the attendance of students in a classroom with some time interval in a day.  All the entries are automatically saved live in a excel sheet .

The algorithm first trains using the images provided to it and records the encodings related to every face . When frames from webcam are fed , it detects all the faces, project them in the correct perspective , encodes them and sees which person has the closest measurements to our face’s measurements. This is the match ! The name of the person is nothing but the filename of that image previously used in training the model . These features are provided in the following api : https://github.com/ageitgey/face_recognition#face-recognition .

## Scanning the image of the person whose attendance is to be marked using webcam

![result_webcam](https://user-images.githubusercontent.com/63535003/118929976-cfd97280-b962-11eb-9f60-42c44a0f33b7.png)

## Name and Time are saved automatically live in an excel sheet

![result_excel](https://user-images.githubusercontent.com/63535003/118928029-4b85f000-b960-11eb-8760-6bb93336f053.png)
