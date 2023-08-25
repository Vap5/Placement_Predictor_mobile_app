import numpy as np
import pandas as pd 
import pickle as pe
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

db=pd.read_csv('D:\kaggle datasets\deploy-ml-model-as-android-app-main\students_placement.csv')
#print(db.shape)
#print(db.sample(11))

x=db.drop(columns=['placed'])
y=db['placed']

#train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
x_train2=x_train.to_numpy()
x_test2=x_test.to_numpy()
#Random forest
rfc=RandomForestClassifier()
rfc.fit(x_train2,y_train)
y_pred=rfc.predict(x_test2)
accuracy_score(y_test,y_pred)
print(accuracy_score(y_test,y_pred))

#Kneighbors
#knc=KNeighborsClassifier(n_neighbors=5)
#knc.fit(x_train,y_train)
#y_pred2=knc.predict(x_test)
#accuracy_score(y_test,y_pred2)
#print(accuracy_score(y_test,y_pred2))

#pickle
pe.dump(rfc,open('placement_predict8.pkl','wb'))

