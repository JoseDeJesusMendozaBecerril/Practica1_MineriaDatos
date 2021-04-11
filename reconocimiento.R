#1.Analisis Exploratorio Dataset Water.csv

data = read.table("water.csv", header=TRUE, sep=",")
data2 = subset(data, select=c(2,3))


#a. Explorar los datos para buscar errores
summary(data)

#b. Generar un resumen con datos de estadistica descriptiva del data set
resumen = summary(data)
write.table(resumen, "resumenWater.txt", sep="\t")

#c. Generar los boxplots correspondientes para analizar el comportamiento de los datos, buscar outliers.
boxplot(data$ï..T_degC, ylab="T_degC", main="boxplot T_degC")
    
    # Guardar boxplot en archivo PNG
    png(file="T_degCBoxplot.png")
    boxplot(data$ï..T_degC, ylab="T_degCBoxplot", main="boxplot T_degC")
    dev.off()

boxplot(data$Salnty, ylab="Salnty", main="boxplot Salnty")
    # Guardar boxplot en archivo PNG
    png(file="SalntyBoxplot.png")
    boxplot(data$Salnty, ylab="SalntyBoxplot", main="Salnty boxplot")
    dev.off()

#d. Generar grafica de disperción
data$colour = "blue"
plot(x=data$ï..T_degC, y=data$Salnty, col=data$colour, xlab="T_degC", ylab="Salnty", main="T_degC vs Salnty")

png(file="T_degC_vs_Salnty.png")
plot(x=data$ï..T_degC, y=data$Salnty, col=data$colour, xlab="T_degC", ylab="Salnty", main="T_degC vs Salnty")
dev.off()




#1.Analisis Exploratorio Dataset mtcars.txt

data2 = read.table("mtcars.txt", header=TRUE)
dataCars = subset(data2, select=c(4,5,7))
dataCars



#a. Explorar los datos para buscar errores
summary(dataCars)

#b. Generar un resumen con datos de estadistica descriptiva del data set
resumenCars = summary(dataCars)
write.table(resumenCars, "resumenCars.txt", sep="\t")

#c. Generar los boxplots correspondientes para analizar el comportamiento de los datos, buscar outliers.

# Guardar boxplot en archivo PNG
    png(file="dispBoxplot.png")
    boxplot(dataCars$disp, ylab="dispBoxplot", main="boxplot disp")
    dev.off()

    png(file="hpBoxplot.png")
    boxplot(dataCars$hp, ylab="hpBoxplot", main="boxplot hp")
    dev.off()

    png(file="wtBoxplot.png")
    boxplot(dataCars$wt, ylab="wtBoxplot", main="boxplot wt")
    dev.off()

#d. Generar grafica de disperción
dataCars$colour = "green"
plot(x=dataCars$disp, y=dataCars$hp, col=dataCars$colour, xlab="disp", ylab="hp", main="disp vs hp")

dataCars$colour = "red"
plot(x=dataCars$wt, y=dataCars$hp, col=dataCars$colour, xlab="wt", ylab="hp", main="wt vs hp")


png(file="disp_vs_hp.png")
plot(x=dataCars$disp, y=dataCars$hp, col=dataCars$colour, xlab="disp", ylab="hp", main="disp vs hp")
dev.off()

png(file="wt_vs_hp.png")
plot(x=dataCars$wt, y=dataCars$hp, col=dataCars$colour, xlab="wt", ylab="hp", main="wt vs hp")
dev.off()