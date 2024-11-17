library(dplyr)
n_obs<-c()
# see some basic informations
for(year in c(2008)){
  print(year)
  df <- read.csv(paste("RealData/Airline/data/", year,'.csv',sep=''))
  df$subj <- paste(df$Month*100 + df$DayofMonth, df$Origin)
  group <- group_by(df, subj)
  smr <- summarise(group, count = n())
  n_obs <- c(n_obs, list(smr$count))
}
sum(unlist(n_obs)>=50&unlist(n_obs)<=100)/365/21
rm(list=ls())

for(year in c(2008)){
  df <- read.csv(paste("RealData/Airline/data/",year,'.csv',sep=''))
  remain = c('Year', 'Month', 'DayofMonth','DepTime','Origin',
             'CRSDepTime','DepDelay')
  df <- df[,remain]
  df <- df[!is.na(df$DepDelay),]
  df$subj <- paste(df$Month*100 + df$DayofMonth, df$Origin)
  df$Delayed <- df$DepDelay>15
  group <- group_by(df,subj)
  smr <- summarise(group, N_obs = n(), N_delay = sum(Delayed))
  smr$DailyDelayRate <- smr$N_delay/smr$N_obs
  save(df,smr,file=paste(year,'_ofda.Rdata',sep=''))
  print(year)
}
rm(list=ls())


# Loop through each year to load and process data (we only use 2008)
year = 2008
set.seed(123)
for(year in c(2008)){
  if(year==2001){set.seed(123)}
  load(paste(year,'_ofda.Rdata',sep=''))
  # Filter data for subjects with at least 50 observations
  proper_subj <- smr$subj[smr$N_obs>=50]
  df <- df[df$subj%in%proper_subj,]
  df <- df[df$DepTime>=600 & df$DepTime<=2300,]
  df$DepTime <- df$DepTime%/%100*60 + df$DepTime%%100
  df <- df[df$DepDelay>=-10 & df$DepDelay<=300,]
  # Normalize DepTime and DepDelay
  df$DepDelay <- (df$DepDelay+10)/310
  df$DepTime <- (df$DepTime - 6*60)/(23*60 - 6*60)
  # Generate unique dates
  df$dates <- df$Month*100+df$DayofMonth
  dates <- unique(df$dates)
  df <- df[df$dates<max(dates),]
  if(year==1989){df[df$dates<=107,'dates'] <- 107; dates <- dates[dates>=107]}
  dates <- sort(dates)
  # Initialize lists to store time, delay and delay rate
  data <- c()
  data <- vector('list', 3) 
  names(data) <- c('t', 'y', 'z')
  for (i in 1:(length(dates)-1)){
    data_ <- df[df$dates == dates[i],]
    t <- c(); y<-c(); z<-c()
    subjs <- unique(data_$subj)
    for(j in subjs){
      TmrrOrigin <- paste(dates[i+1],strsplit(j,' ')[[1]][2])
      if(TmrrOrigin%in%smr$subj){
        t1 <- data_[data_$subj==j, 'DepTime']
        y1 <- data_[data_$subj==j, 'DepDelay']
        m_obs <- 50  ## sparsify data
        if (m_obs<length(t1)){
          idx <- sample(1:length(t1), min(m_obs,length(t1)))
          t1 <- t1[idx]; y1<-y1[idx]
          t <- c(t, list(t1))
          y <- c(y, list(y1))
          z <- c(z, as.numeric(smr[smr$subj==TmrrOrigin,'DailyDelayRate']))
        }
      }
    }
    n_sub <- 60 # aviod data imbalance
    idx <- sample(1:length(t), min(n_sub,length(t)))
    data$t <- c(data$t,t[idx])
    data$y <- c(data$y,y[idx])
    data$z <- c(data$z,z[idx])
    print(paste('year',year,'-',dates[i]))
  }
  # Save the final processed data
  save(data, file=paste('RealData/Airline/data/datahao/data',year,'.Rdata',sep=''))
}