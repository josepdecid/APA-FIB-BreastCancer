PATH = 'Dropbox/APA/Practica/APA-FIB-BreastCancer/dataset/data.csv'
data <- read.csv(file = PATH, header = TRUE, sep = ",", dec = ".", allowEscapes = TRUE )

summary(data[['radius_mean']])


