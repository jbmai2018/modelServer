from django.contrib.auth.models import User
from django.contrib.auth import authenticate
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rest_framework.exceptions import ParseError
from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import FileSerializer
import productionPlanning.analysis as analysis
import productionPlanning.dbserver as db

# from captureImage.frameAcq import *

class predict(generics.RetrieveUpdateDestroyAPIView):
    # authentication_classes = (TokenAuthentication,)
    # permission_classes = (IsAuthenticated,)
    @csrf_exempt
    def get(self, request):
        answer = analysis.get_result()
        dataToSend = {
            "result" : answer
        }
        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')
    @csrf_exempt
    def post(self, request):
        crPath = request.data['crPath']
        answer = analysis.get_result(crPath)
        print(answer)
        dataToSend = {
            "result" : answer
        }

        return JsonResponse(
            dataToSend,
            safe=False,content_type='application/json')


class FileView(APIView):
    parser_classes = (MultiPartParser,FormParser)

    @csrf_exempt
    def post(self,request,*args,**kwargs):
        file_serializer = FileSerializer(data=request.data)
        if file_serializer.is_valid():
            file_serializer.save()
            print(file_serializer.data)
            db.store_in_db(file_serializer.data)
            return Response(file_serializer.data,status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors,status=status.HTTP_400_BAD_REQUEST)
