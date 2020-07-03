import cv2
import os
from datetime import datetime
from datetime import date

cameraList = [
    ("rtsp://admin:password%40123@192.1.2.131:554/live/0/MAIN",'1'),
    ("rtsp://admin:password%40123@192.1.2.132:554/live/0/MAIN",'2'),
    ("rtsp://admin:password%40123@192.1.2.133:554/live/0/MAIN",'3'),
    ("rtsp://admin:password%40123@192.1.2.134:554/live/0/MAIN",'4'),
    ("rtsp://admin:password%40123@192.1.2.7:554/live/0/MAIN",'5')
]

videoCaptureObjectArray = []
for cam in cameraList:
    videoCaptureObjectArray.append((cv2.VideoCapture(cam[0]),cam[1]))

today = date.today()
d1 = today.strftime("%d-%m-%Y")
qualityProjectPath = "/home/jbmai/qualityProject/JBMQualityproject"
def acquireFrames(data):
    partId = data['partId']
    setupType = str(data['cameraSetup'])
    tempImagePaths = []
    for cam in cameraList:
        if cam[1] in data['imagesToAcquire'] :
            videoCaptureObject = cv2.VideoCapture(cam[0])
            result = True
            while(result):
                for x in range(10):
                    ret,frame = videoCaptureObject.read()

                pathAppend = "/data/qualityFrames/" + d1 + "/" + setupType + "_camera/" + partId + "/" + cam[1] + "/"
                directory = qualityProjectPath + pathAppend

                my_date = datetime.now()
                fileName = my_date.strftime('%Y-%m-%dT%H:%M:%S.%f%z') + ".jpg"

                if not os.path.exists(directory):
                    os.makedirs(directory)

                print("Wrting Image")
                cv2.imwrite((directory + fileName),frame)
                result = False
                tempImagePaths.append({
                'qualityProjectPath' : qualityProjectPath,
                'relativePath' : pathAppend + fileName,
                'path' : pathAppend + fileName,
                'camera' : cam[1]
                })
            videoCaptureObject.release()
            cv2.destroyAllWindows()
    print("Captured")
    return tempImagePaths
