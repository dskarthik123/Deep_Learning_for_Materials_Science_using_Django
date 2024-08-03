from django.urls import path
from .views import predict, home

urlpatterns = [
    path('predict/', predict, name='predict'),
    path('', home, name='home'),
]
