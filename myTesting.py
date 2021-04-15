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
from matplotlib import markers

import seaborn as sns

filename1 = 'wate2r.csv'
filename2 = 'mtcars.txt'


3
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

    print("Input n for validation n-fold cross validation")
    n = int(input(">>"))
    kf=KFold(n_splits=n,shuffle=True,random_state=402 )

    #Se instancia modelo de regresion lineal
    regr=linear_model.LinearRegression()

    for values_x,values_y in kf.split(X):
        #Se obtienen los datos a patir de los arreglos particion por k-fold
        X_train,X_test = X[values_x],X[values_y]
        Y_train,Y_test = Y[values_x],Y[values_y]
        #Se obtienen ajusta la estructura de los datos para ser recibidos por 
        Y_train,Y_test = Y_train.reshape(-1,1),Y_test.reshape(-1,1)
        print(values_x,values_y)
        #Se entrena al modelo
        
        regr.fit(X_train,Y_train)
        #Se hace una preccion con el modelo
        y_pred=regr.predict(X_train)

    #Se imprime la ultima prediccion obtenida
    print("Datos predecidos\n",y_pred,"\nDatos reales\n",Y_test)
    #Se obtiene la bondad del modelo
    print("Bondad del modelo", regr.score(X_train,Y_train) )
    #Se imprime el error cuadratico medio
    print("Error cuadratico medio", mean_squared_error(Y_train,y_pred) )

    #Se da estructura a los datos de test set para poder graficarlos
    xx_pred, xx1_pred = np.meshgrid(X_test[:,0], X_test[:,1])
    model_viz = np.array([xx_pred.flatten(), xx1_pred.flatten()]).T
    #Se obtiene el plano para poder ser graficado a partir de data det test
    yy_predicted = regr.predict(model_viz)

    #Se define nueva grafica 3d
    fig =plt.figure()
    ax= fig.add_subplot(111,projection='3d')

    #se grafican los datos del data test
    ax.scatter(X_test[:,0],X_test[:,1],Y_test[:,0],c='red',marker='o',alpha=0.6 )
    #se grafican los datos del plano de acuerdo al modelo dado
    ax.scatter3D(xx_pred.flatten(), xx1_pred.flatten(), yy_predicted,color='#58A4C8', marker=markers.TICKRIGHT, s=100 )

    plt.show()
    plt.clf()

    print("shape",Y_train.shape, y_pred.shape)
    residuos_test  = Y_train - y_pred
    
    plt.title('Grafica de Residuos')
    plt.hist(residuos_test, bins=10, alpha=0.5, edgecolor = 'black',  linewidth=1)
    plt.grid(True)
    plt.show()
    plt.clf()
    

    
main()