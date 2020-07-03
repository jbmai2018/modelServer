from django.conf.urls import url, include
from rest_framework.authtoken import views
from nltkServer.views import *

app_name = 'nltkServer'

urlpatterns = [
    url(r'comments/', getPrediction.as_view()),
]
