import pandas as pd
import numpy as np
from sklearn import linear_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

filename1 = 'water.csv'
filename2 = 'mtcars.txt'



def main():
    '''
    # ----------------------------- Lectura de datos Water -------------------
    water = pd.read_csv(filename1, header=0)
    print(water.shape)
    print (water.head(645))
    #print(data.columns)

    #print(data['T_degC'])
    #print(data['Salnty;;'])
    
    print("\nMedias antes de modificar datos vacios")
    mediaT_degC = water['T_degC'].mean()
    mediaSalnty = water['Salnty'].mean()
    print('La media de T_degC es' , mediaT_degC)
    print('La media de Salnty es' , mediaSalnty)
    
    #2. Limpieza de datos - Reparacion de datos
    water['T_degC'] = water['T_degC'].replace(np.nan,mediaT_degC)
    water['Salnty'] = water['Salnty'].replace(np.nan,mediaSalnty)
    
    print (water.head(645))
    
    print("\nMedias despues de modificar datos vacios")
    mediaT_degC = water['T_degC'].mean()
    mediaSalnty = water['Salnty'].mean()
    print('La media de T_degC es' , mediaT_degC)
    print('La media de Salnty es' , mediaSalnty)
    
    # 3. Regresion Lineal

    #Seleccionamos X y Y
    X = water['Salnty'] #Entrada
    y = water['T_degC'] #Salida

    #plt.scatter(X,y)
    #plt.xlabel('T_degC')
    #plt.ylabel('Salnty')
    #plt.show()

    #Separar los datos en entrenamiento y prueba
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    
    
    #Definimos algoritmo a utilizar
    lr = linear_model.LinearRegression()

    X_train = X_train.values.reshape(-1,1)
    y_train = y_train.values.reshape(-1,1)
    X_test = X_test.values.reshape(-1,1)
    y_test = y_test.values.reshape(-1,1)
    
    print("This is X_train",X_train)
    print("This is y_train",y_train)
    

    #Entreno el modelo
    lr = lr.fit(X_train,y_train)
    
    #Realizo una prediccion
    Y_pred = lr.predict(X_test) #Uso valores de test set

    #Graficamos los datos junto con el modelo
    plt.scatter(X_test,y_test)
    plt.plot(X_test,Y_pred,color='red',linewidth=3)
    plt.title('Regresion Lineal Simple')
    plt.xlabel('Salnty')
    plt.ylabel('T_degC')
    plt.show()
    

    #4. Evaluacion
    #Debido a que el dataset es muy grande se utiliza la metodologia de validacion simple 
    print("Precision del modelo")
    print(lr.score(X_train,y_train))

    
    mse = mean_squared_error(y_test,Y_pred)
    print("Error cuadratico medio",mse)

    #5. Grafica de residuos
    Y_predtrain = lr.predict(X_train) #Uso valores de training set


    print("y_train" , y_train[10])
    print("Y_pred", Y_predtrain[10])
    

    residuos_test  = y_train - Y_predtrain
    sns.histplot(data = residuos_test)
    plt.show() #Muestra grafica de distribucion


'''
#-------------------------------------------------------------------------------------------------------------------

    #Lectura de datos Cars  (disp,wt) entrada (hp) salida
    cars = pd.read_csv(filename2,header=0,sep=" ")
    print(cars.shape)
    print (cars.head(32))
    print(cars.columns)

    #print(cars['disp']) #valores de entrada
    #print(cars['wt'])
    #print(cars['hp']) # Valor de salida



    #Limpieza de datos
    print("\nMedias antes de modificar datos vacios")
    mediaDisp = cars['disp'].mean()
    mediaWt = cars['wt'].mean()
    mediaHp = cars['hp'].mean()
    print('La media de disp es' , mediaDisp)
    print('La media de wt es' , mediaWt)
    print('La media de hp es' , mediaHp)
    
    #2. Limpieza de datos - Reparacion de datos no necesaria debido a que no existe NaN
    

    # 3. Regresion Lineal

    

main()