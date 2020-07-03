from django.conf.urls import url, include
from rest_framework.authtoken import views
from isolationForest.views import *

app_name = 'isolationForest'

urlpatterns = [
    url(r'predict', predict.as_view()),
]
