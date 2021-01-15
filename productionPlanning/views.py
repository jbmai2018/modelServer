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
import productionPlanning.analysis as analysis

# from captureImage.frameAcq import *

class predict(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    def get(self, request):
        # answer = analysis.get_result(requestData)
        dataToSend = {
            "result" : "ass1"
        }
        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')

    def post(self, request):
        crPath = request.data['crPath']
        emergPath = request.data['emergPath']
        answer = analysis.get_result(crPath)
        print(answer)
        dataToSend = {
            "result" : answer
        }

        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
