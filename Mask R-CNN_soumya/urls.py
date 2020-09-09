from django.conf.urls import url, include
# from rest_framework.authtoken import views
from Maskrcnn.views import *
app_name = 'Maskrcnn'

urlpatterns = [
    url(r'predict', predict.as_view()),
]


# ##Sahil, please check and confirm if everything is right. Thanks
