from django.contrib import admin
from django.urls import path
from covid19 import views

urlpatterns = [
    path('',views.home, name='base'),
    path('aboutus', views.aboutus, name='aboutus'),
    path('contactus', views.contactus, name='contactus'),
    path('covid', views.covid, name='covid'),
    path('brain_stroke',views.brain_stroke,name ='brain_stroke'),
    path('services', views.services, name='services'),
    path('login', views.loginPage, name="login" ),
    path('register',views.registerPage, name="register" ),
    path('covidTest',views.covidTest, name="covidTest" ),
    path('brain_stroke_result',views.brain_stroke_result,name = 'brain_stroke_result')
 

   # path('predictImage',views.predictImage,name = 'predictImage'),
] 
#urlpatterns+= static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
#urlpatterns+= static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)