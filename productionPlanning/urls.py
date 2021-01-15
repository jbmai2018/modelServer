from django.conf.urls import url, include
from rest_framework.authtoken import views
from productionPlanning.views import *

app_name = 'productionPlanning'

urlpatterns = [
    url(r'predict', predict.as_view()),
]
