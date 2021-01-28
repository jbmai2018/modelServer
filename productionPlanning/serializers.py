from rest_framework import serializers
from .models import File


# Serializers define the API representation
class FileSerializer(serializers.ModelSerializer):
    class Meta():
        model = File
        fields = ('file','remark','timestamp')
