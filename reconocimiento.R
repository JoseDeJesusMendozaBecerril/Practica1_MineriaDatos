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