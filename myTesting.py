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

    #Lectura de datos mtCars

    mtCars = pd.read_csv(filename2,sep=" ",header=0)
    mtCars = mtCars[['disp','hp','wt']]
    df=pd.DataFrame(mtCars)

    #Lectura de datos variables de entrada
    disp = df['disp']
    wt = df['wt']
    X=np.array([disp,wt]).T

    #Lectura de datos variables de salida
    hp = df['hp']
    Y=np.array(hp).T

    

    kf=KFold(n_splits=2,shuffle=True,random_state=2)
    regr=linear_model.LinearRegression()

    for values_x,values_y in kf.split(X):

        X_train,X_test = X[values_x],X[values_y]
        Y_train,Y_test = Y[values_x],Y[values_y]

        Y_train,Y_test = Y_train.reshape(-1,1),Y_test.reshape(-1,1)
        regr.fit(X_train,Y_train)
        y_pred=regr.predict(X_test)

        print(y_pred,Y_test)
        print(regr.score(X_train,Y_train) )
        print(mean_squared_error(Y_test,y_pred) )

        xx_pred, yy_pred = np.meshgrid(X_test[:,0], X_test[:,1])
        model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T
        predicted = regr.predict(model_viz)

        fig =plt.figure()
        ax= fig.add_subplot(111,projection='3d')
        ax.scatter(X_test[:,0],X_test[:,1],Y_test[:,0],c='red',marker='o',alpha=0.5 )
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0),marker='o', s=25, edgecolor='#70b3f0')

        plt.show()

    
main()