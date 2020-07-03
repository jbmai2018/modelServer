from django.conf.urls import url, include
# from rest_framework.authtoken import views
from autoencoder_mayank.views import *

app_name = 'autoencoder_mayank'

urlpatterns = [
    url(r'predict', predict.as_view()),
]
