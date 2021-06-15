from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
from .forms import CreateUserForm
from django.core.files.storage import default_storage
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import Contact
from django.utils.datastructures import MultiValueDictKeyError


import torch 
import torch.nn as nn 
from torch.utils.data import Dataset , DataLoader 
import albumentations as a 
import cv2 
#import matplotlib.pyplot as plt 
import timm 
import numpy as np
from torchvision import transforms
import pickle
from .machine_learning.stroke_prediction import stroke_model
from .deep_learning_models.Covid_pne import FeatureExtractor
from .deep_learning_models.Covid_pne import dataset
from .deep_learning_models.Covid_pne import ModelOutputs
from .deep_learning_models.Covid_pne import GradCam
from .deep_learning_models.Covid_pne import Prediction


def home(request):
    return render(request,"base.html")
    
def aboutus(request):
    return render(request, "about.html")

def contactus(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        subject = request.POST['subject']
        message = request.POST['message']
        if len(name)< 2 or len(email) < 3 or len(subject) < 3 or len(message) < 3:
            messages.error(request , "Please Fill the form correctly")
        else:
            contact  = Contact(name = name, email = email, subject = subject,content = message)
            contact.save()
            messages.success(request , "Message has been sent successfully !!!")
    return render(request,"contact.html")
def covid(request):
    return render(request,"covid.html")
def brain_stroke(request):
    return render(request,'brain_stroke.html')
def brain_stroke_result(request):
    return render(request,'brain_stroke_result.html')

def services(request):
    return render(request,"services.html")


def loginPage(request):
    if request.method == 'POST':
        username =   request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username = username, password = password)

        if user is not None:
            login(request, user)
            return redirect('base')

    context = {}
    return render(request, 'login.html', context)


def registerPage(request):
    form  = CreateUserForm()

    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user = form.cleaned_data.get('username')
            messages.success(request, 'Account is created for ' + user)
            return redirect('login')
    context = {'form': form}
    return render(request, 'register.html', context)



def covidTest(request):
    if request.method == "POST":
      #  file = request.FILES["imageFile"]
        try:
            file = request.FILES["imageFile"]
        except MultiValueDictKeyError:
            return render(request, "covid.html")
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        print(file_url) 
       
        path = [file_url]
        print(f'path is :{path}')
        obj = Prediction(path)
        output = obj.show_prediction()  
        print(output)
        
        return render(request, 'covid.html', {"predictions": output})
    else:
        return render(request, "covid.html")


def brain_stroke_pred(gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,smoking_status,height,weight):
    height = height/100
    bmi = weight/height 
    model  = stroke_model()
    pred = model.predict(np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,bmi,smoking_status]]))

    if pred[0] ==  0:
        return 'yes'
    elif pred[0] == 1:
        return 'no'
    else:
        return 'got some error'


def brain_stroke_result(request):
    Gender = int(request.GET['Gender'])
    age = int(request.GET['age'])
    hypertension= int(request.GET['hypertension'])
    heart_disease = int(request.GET['heart_disease'])
    ever_married = int(request.GET['ever_married'])
    work_type = int(request.GET['work_type'])
    residence_type = int(request.GET['residence_type'])
    avg_glucose_level = int(request.GET['avg_glucose_level'])
    smoking_status = int(request.GET['smoking_status'])
    height = int(request.GET['height'])
    weight= int(request.GET['weight'])

    result = brain_stroke_pred(Gender,age,hypertension,heart_disease,ever_married,work_type,residence_type,avg_glucose_level,smoking_status,height,weight)
    
    return render(request,'brain_stroke_result.html',{'result':result})


