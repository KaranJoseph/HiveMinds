import numpy as np 
import pandas as pd 
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
#import matplotlib.pyplot as plt

file_location='D:\\Karan\\R_Module_Day_5.2_Data_Case_Study_Loss_Given_Default.csv' ## Change File Location Here.
data=pd.read_csv(file_location,sep=',') 

data['Number of Vehicles']=data['Number of Vehicles'].astype('object')

'''
01. Data interpretation and insight genertation

'''

def age(x):
    if(x<=20):
        return '<=20'
    elif x<=40:
        return '21-40'          ##Dividing Age into classes
    elif x<=60:
        return '41-60'
    else:
        return '>60'
    
data['ClassAge']=data['Age'].apply(age)    

def exp(x):
    if(x<10):
        return 'Less than 10 Years'
    elif x<20:
        return '10-19 Years'
    elif x<30:
        return '20-29 Years'       ##Making Years of Experience into classes.
    elif x<40:
        return '30-39 Years'
    elif x<50:
        return '40-59 Years'
    else:
        return 'More than 60 Years'
    
data['ClassExp']=data['Years of Experience'].apply(exp)

model1=pd.pivot_table(data,values='Losses in Thousands',
    index=['ClassAge','ClassExp','Number of Vehicles','Gender','Married'],
    aggfunc=lambda x:round((100*np.sum(x)/data['Losses in Thousands'].sum()),2))   ##Finding percentage loss for each combination of categories    

model2=pd.pivot_table(data,values='Ac_No',
        index=['ClassAge','ClassExp','Number of Vehicles','Gender','Married'],
        aggfunc='count')    
model2.rename(columns={'Ac_No':'Count'},inplace=True)  ## Finding count of columns in each categories

model=pd.merge(model1,model2,left_index=True,right_index=True)

model.sort_values(by=['Losses in Thousands','Count'],ascending=False,inplace=True)

model.reset_index(level=['ClassAge','ClassExp','Number of Vehicles','Gender','Married'],inplace=True)

print("Top 5 Loss making categories are: \n {}".format(model[['ClassAge','ClassExp',
      'Number of Vehicles','Gender','Married']].head(5)))
print("\n")
print("5 Least Loss making categories are:  \n {}".format(model[['ClassAge','ClassExp',
      'Number of Vehicles','Gender','Married']].tail(5)))

'''02. Linear regression and Loss Prediction'''

data.Gender=data['Gender'].apply(lambda x:1 if x=='M' else 2)
data.Married=data.Married.apply(lambda x:1 if x=='Married' else 2)

reg=linear_model.LinearRegression()
reg.fit(data[['Age','Years of Experience','Number of Vehicles','Gender','Married']]
                                                        ,data['Losses in Thousands'])

intercept=reg.intercept_
coef=np.array(reg.coef_)


'''03. Polynomial Regression'''

poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(data[['Age','Years of Experience',
                                    'Number of Vehicles','Gender','Married']])

poly_reg.fit(X_poly,data['Losses in Thousands'])
reg2=linear_model.LinearRegression()
reg2.fit(X_poly,data['Losses in Thousands'])

counter = 0
while counter!=1:
    X=[0,1,2,3,4]
    X[0]=int(input('Enter age \t'))
    X[1]=float(input('Enter Years of Experience \t:'))
    X[2]=int(input('Enter Number of Vehicles \t:'))
    X[3]=input('Enter Gender, M for male and F for Female \t:')
    X[4]=input('Enter Marrital Status, Married or Single \t:')
    
    if X[3]=='M':
        X[3]=1
    elif X[3]=='F':
        X[3]=2
    else:
        print('Wrong Input for Gender, Run Code Again')
        break
    
    if X[4]=='Married':
        X[4]=1
    elif X[4]=='Single':
        X[4]=2
    else:
        print('Wrong Input for Marrital Status, Run Code Again') 
        break
        
    
    print('Predicted Loss for the inputs by linear regression is: \n {}'.format(list(reg.predict([X]))))
    print('Predicted loss for the inputs by polynomial regression is: \n {}'.format(list(reg2.predict(poly_reg.fit_transform([X])))))
    counter=input('Enter 1 to exit loop \t:')
      

    
    



