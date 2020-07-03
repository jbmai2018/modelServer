from django.shortcuts import render
import cv2
# Create your views here.
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
# from auto.frameAcq import *

class predict(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def post(self, request):
        requestData = request.data['path']
        print("bodyyy",requestData)
        dataToSend = "hello returned"
        path = '"'  + requestData+ '"'
        print(path)
        img = cv2.imread(requestData)
        answer = mayank.input(img)
        print("Output -----"+str(answer))
        # image = request.FILES['image']
        # print(image)
        # cv2.imshow("nn",img)
        # cv2.waitKey(0)
        # dataToSend = "hello returned"
        # print(dataToSend)
        return JsonResponse(
            str(answer),
            safe=False,content_type='application/json')
