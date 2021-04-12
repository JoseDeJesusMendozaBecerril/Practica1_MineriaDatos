import pandas as pd
import numpy as np
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

filename1 = 'wate2r.csv'
filename2 = 'mtcars.txt'



def main():
    print("Hola Mundo")
    #Lectura de datos mtCars
    mtCars = pd.read_csv(filename2,sep=" ",header=0)
    mtCars = mtCars[['disp','hp','wt']]
    df=pd.DataFrame(mtCars)

    disp = df['disp']
    hp = df['hp']
    wt = df['wt']
    
    X=np.array([disp,wt]).T
    Y=np.array(hp).T

    print(Y)

    kf=KFold(n_splits=2,shuffle=True,random_state=2)
    regr=linear_model.LinearRegression()

    for values_x,values_y in kf.split(X):
        #print(df.iloc[values_x],df.iloc[values_y])
        #print(values_x,values_y)
        X_train,X_test = X[values_x],X[values_y]
        Y_train,Y_test = Y[values_x],Y[values_y]
        #X_train,X_test = X_train.reshape(-1,1),X_test.reshape(-1,1)
        Y_train,Y_test = Y_train.reshape(-1,1),Y_test.reshape(-1,1)
        regr.fit(X_train,Y_train)
        y_pred=regr.predict(X_test)

        print(y_pred,Y_test)
        print(mean_squared_error(Y_test,y_pred) )

    




    """ mtCars = mtCars[['disp','hp','wt']]
    
    print(mtCars.shape)
    print (mtCars.head(5))

    disp = mtCars['disp'].values
    hp = mtCars['hp'].values
    wt = mtCars['wt'].values
    
    X=np.array([disp,wt]).T
    Y=np.array(hp)

    reg = linear_model.LinearRegression()
    reg=reg.fit(X,Y)
    Y_pred=reg.predict(X)

    error = np.sqrt(mean_squared_error(Y,Y_pred))
    r2=reg.score(X,Y)

    print("error :",error)
    print("r cuadrada", r2)
    print("los coeficientes son: \n",reg.coef_)

 """
main()