from django.conf.urls import url, include
from rest_framework.authtoken import views
from captureImage.views import *

app_name = 'captureImage'

urlpatterns = [
    url(r'capture', captureImage.as_view()),
]
