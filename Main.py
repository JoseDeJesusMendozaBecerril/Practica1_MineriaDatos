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
from matplotlib import markers

filename1 = 'water.csv'
filename2 = 'mtcars.txt'



def main():

    # ----------------------------- Lectura de datos Water - Regresion Lineal Simple -------------------
    water = pd.read_csv(filename1, header=0)
    #print(water.shape)
    #print (water.head(645))
    #print(data.columns)

    #print(data['T_degC'])
    #print(data['Salnty;;'])
    print("******************** Modelo RLS ********************")
    print("\nMedias antes de modificar datos vacios")
    mediaT_degC = water['T_degC'].mean()
    mediaSalnty = water['Salnty'].mean()
    print('La media de T_degC es' , mediaT_degC)
    print('La media de Salnty es' , mediaSalnty)
    
    #2. Limpieza de datos - Reparacion de datos
    water['T_degC'] = water['T_degC'].replace(np.nan,mediaT_degC)
    water['Salnty'] = water['Salnty'].replace(np.nan,mediaSalnty)
    
    #print (water.head(645))
    
    print("\nMedias despues de modificar datos vacios")
    mediaT_degC = water['T_degC'].mean()
    mediaSalnty = water['Salnty'].mean()
    print('La media de T_degC es' , mediaT_degC)
    print('La media de Salnty es' , mediaSalnty)
    
    # 3. Regresion Lineal

    #Seleccionamos X y Y
    X = water['Salnty'] #Entrada
    y = water['T_degC'] #Salida

    
    #Separar los datos en entrenamiento y prueba
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    
    
    #Definimos algoritmo a utilizar
    lr = linear_model.LinearRegression()

    X_train = X_train.values.reshape(-1,1)
    y_train = y_train.values.reshape(-1,1)
    X_test = X_test.values.reshape(-1,1)
    y_test = y_test.values.reshape(-1,1)
    
    #print("This is X_train",X_train)
    #print("This is y_train",y_train)
    
    
    #Entreno el modelo
    lr = lr.fit(X_train,y_train)
    
    #Realizo una prediccion
    Y_pred = lr.predict(X_test) #Uso valores de test set

    fig, ax = plt.subplots(3)

    #Graficamos los datos junto con el modelo
    ax[0].scatter(X_test,y_test)
    ax[0].plot(X_test,Y_pred,color='red',linewidth=3)
    ax[0].set_title('Regresion Lineal Simple')
    ax[0].set_xlabel('Salnty')
    ax[0].set_ylabel('T_degC')
  
    
 
    #4. Evaluacion
    #Debido a que el dataset es muy grande se utiliza la metodologia de validacion simple 
    
    print("Bondad:" , lr.score(X_train,y_train))

    
    mse = mean_squared_error(y_test,Y_pred)
    print("Error cuadratico medio",mse)

    #5. Grafica de residuos
    Y_predtrain = lr.predict(X_train) #Uso valores de training set


    #print("y_train" , y_train[10])
    #print("Y_pred", Y_predtrain[10])
    
    residuos_test  = y_train - Y_predtrain

# ----------------------------- Lectura de datos Water - Regresion Lineal Ponderada -------------------    

    #Calculasmos W
    m_wlr = 1/residuos_test**2
    #print(m_wlr[:,0] )
    #Entrenamos el modelo
    lr = lr.fit(X_train,y_train,sample_weight=m_wlr[:,0])
    #hacemos una predccion
    Y_pred = lr.predict(X_test)

    ax[1].scatter(X_test,y_test)
    ax[1].plot(X_test,Y_pred,color='green',linewidth=3)
    ax[1].set_title('Regresion Lineal ponderda')
    ax[1].set_xlabel('Salnty')
    ax[1].set_ylabel('T_degC')

    print("******************** modelo RLP ********************")
    print("Bondad: ",lr.score(X_train,y_train))

    
    mse = mean_squared_error(y_test,Y_pred)
    print("Error cuadratico medio RLP",mse)

    sns.histplot(data = residuos_test)

    plt.show()


  
    

    


#------------------------------------------------------Regresion Lineal Multiple -------------------------------------------------------------
    print("******************** modelo RLM ********************\n")
    #Lectura de datos Cars  (disp,wt) entrada (hp) salida
    mtCars = pd.read_csv(filename2,sep=" ",header=0)
    mtCars = mtCars[['disp','hp','wt']]
    df=pd.DataFrame(mtCars)

    #Limpieza de datos
    print("\nMedias antes de modificar datos vacios")
    mediaDisp = mtCars['disp'].mean()
    mediaWt = mtCars['wt'].mean()
    mediaHp = mtCars['hp'].mean()
    print('La media de disp es' , mediaDisp)
    print('La media de wt es' , mediaWt)
    print('La media de hp es' , mediaHp)
    
    #2. Limpieza de datos - Reparacion de datos no necesaria debido a que no existe NaN

    # 3. Regresion Lineal

    #Lectura de datos variables de entrada
    disp = df['disp']
    wt = df['wt']
    X=np.array([disp,wt]).T

    #Lectura de datos variables de salida
    hp = df['hp']
    Y=np.array(hp).T

    print("Input n for validation n-fold cross validation")
    n = int(input(">>"))
    kf=KFold(n_splits=n,shuffle=True,random_state=40 )

    #Se instancia modelo de regresion lineal
    regr=linear_model.LinearRegression()

    for values_x,values_y in kf.split(X):
        #Se obtienen los datos a patir de los arreglos particion por k-fold
        X_train,X_test = X[values_x],X[values_y]
        Y_train,Y_test = Y[values_x],Y[values_y]
        #Se obtienen ajusta la estructura de los datos para ser recibidos por 
        Y_train,Y_test = Y_train.reshape(-1,1),Y_test.reshape(-1,1)
        #print(values_x,values_y)
        #Se entrena al modelo
        
        regr.fit(X_train,Y_train)
        #Se hace una preccion con el modelo
        y_pred=regr.predict(X_test)

    #Se imprime la ultima prediccion obtenida
    #print("Datos predecidos\n",y_pred,"\nDatos reales\n",Y_test)
    #Se obtiene la bondad del modelo
    
    print("Bondad:", regr.score(X_train,Y_train) )
    #Se imprime el error cuadratico medio
    print("Error cuadratico medio", mean_squared_error(Y_test,y_pred) )

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
    ax.set_xlabel("disp")
    ax.set_ylabel("wt")
    ax.set_zlabel("hp")
    plt.show()
    plt.clf()
    #Residuos
    y_pred=regr.predict(X_train)
    residuos_test  = Y_train - y_pred
    
    plt.title('Grafica de Residuos')
    plt.hist(residuos_test, bins=10, alpha=0.5, edgecolor = 'black',  linewidth=1)
    plt.grid(True)
    plt.show()
    plt.clf()
    
main()