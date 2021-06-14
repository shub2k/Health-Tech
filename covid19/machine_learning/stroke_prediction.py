import pickle 
import numpy as np

# Loading model
def stroke_model():
    model = pickle.load(open('covid19/machine_learning/cat_boost_stroke/stroke_cat', 'rb'))
    return model
 

dic ={ 
    'Gender':{'Male':1,'Female':0,'Other':2},
    'Hypertension':{'yes':1,'no':0},
    'Heart_disease':{'yes':1,'no':0},
    'Ever_married':{'yes':1,'no':0},
    'Work_type':{'govt_job':0,'Private':2,'self_employed':3,'never_worked':1},
    'Residence_type':{'urban':1,'rural':0},
    'smoking_status':{'formerly_smokes':1,'never_smokes':2,'smokes':3,'unkown':0}
}
gender = 1 # Male = 1, Female = 0 , 2 = others 
Hypertension = 0 # yes = 1 , no = 0
heart_disease = 0 # yes = 1 , no = 0
ever_married = 0 # yes = 1 , no = 0
Work_type = 3 # govt_job = 0 , never_worked = 1, private = 2 , self_employed = 3
Residence_type = 1 # urban = 1 , rural = 0
smoking_status = 0 # unknown = 0 , formerly_smokes = 1 , never_smokes = 2 , smokes = 3
avg_glucose_level = 155
bmi = 35
age = 21
if __name__ == '__main__':
    model = stroke_model()
    pred = model.predict(np.array([[gender,age,Hypertension,heart_disease,ever_married,Work_type ,Residence_type,avg_glucose_level,bmi,smoking_status]]))
    
    if pred[0] ==  0:
        print('Bro model predicted that you will not get brain stroke just eat healthy foods , do meditation and play pubg')
    else:
        print('Bro model predicted that you will be suffer from brain stroke we highly recommend you to check yourself with a doctor')
