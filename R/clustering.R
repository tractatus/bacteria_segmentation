files <- dir('data', full.names = TRUE)

ecoli<-read.table(files[1], sep=',', header=TRUE)
sent <- read.table(files[2], sep=',', header=TRUE)

quartz()
plot(ecoli$Mean, sent$Mean, asp=1, pch=16, cex=0.5, col=rgb(0,0,0, 0.1))
