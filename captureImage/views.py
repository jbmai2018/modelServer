from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.http import JsonResponse

from rest_framework.exceptions import ParseError
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import status

import autoencoder_mayank.test1 as mayank
import isolationForest.prediction as Mukul
from captureImage.frameAcq import *

class captureImage(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def post(self, request):
        requestData = request.data
        dataToSend = {}
        dataToSend['framePaths'] = []
        framePaths = acquireFrames(requestData)

        for frame in framePaths:
            tempFrame = frame
            if(frame['camera'] == '1'):

                path = frame['qualityProjectPath']+frame['relativePath']
                print(path)
                ###################################
                # img = cv2.imread(path)
                answer,answer2 = mayank.input(path)
                send =str(answer)+"------"+str(answer2)
                print("-----------------------------------"+str(answer))
                if float(answer) >= float(5.0):

                    tempFrame['autoencoder_mayank'] = {
                        'ssim' : send ,
                        'okNg' : "OK"
                    }
                    tempFrame['status'] = 'OK'
                else:

                    tempFrame['autoencoder_mayank'] = {
                        'ssim' : answer,
                        'okNg' : "NG"
                    }
                    tempFrame['status'] = 'NG'
                ###################################
                answer = Mukul.get_result(path)
                tempFrame['isolationForest'] = {
                    'ssim' : str(answer[0]),
                    'okNg' : "NG"
                }
                ###################################

                # tempFrame['status'] = 'NG'
                dataToSend['framePaths'].append(tempFrame)
            else:
                tempFrame['status'] = 'NotCalculated'
                dataToSend['framePaths'].append(tempFrame)

        # dataToSend = {array with 1 element ndarray?
        #     "framePaths" : framePaths
        # }use .tolist()its fine nowhahahahaha

        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
