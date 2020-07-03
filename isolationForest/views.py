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
import isolationForest.prediction as Mukul

# from captureImage.frameAcq import *

class predict(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def post(self, request):
        requestData = request.data['path']
        answer = Mukul.get_result(requestData)
        # dataToSend = requestData
        # print("data",requestData)
        # print(requestData['partId'])
        # framePaths = acquireFrames(requestData)
        dataToSend = {
            "result" : str(answer[0])
        }

        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
