from django.conf.urls import url, include
from django.urls import path
from rest_framework.authtoken import views
from rest_framework import routers
from .views import FileView
from productionPlanning.views import *


app_name = 'productionPlanning'

urlpatterns = [
    url(r'predict', predict.as_view()),
    url(r'^upload/$',FileView.as_view(),name='file-upload'),
]
