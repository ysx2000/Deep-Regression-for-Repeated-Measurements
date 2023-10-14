library(argoFloats)
library(oce)

indexAll <- getIndex()
timebeginmon <- c("-06-01 00:00:00 UTC", "-06-11 00:00:00 UTC", 
                  "-06-21 00:00:00 UTC", "-07-01 00:00:00 UTC", 
                  "-07-11 00:00:00 UTC", "-07-21 00:00:00 UTC", 
                  "-08-01 00:00:00 UTC", "-08-11 00:00:00 UTC", 
                  "-08-21 00:00:00 UTC")

timeendmon <- c("-06-01 23:59:59 UTC", "-06-11 23:59:59 UTC", 
                "-06-21 23:59:59 UTC", "-07-01 23:59:59 UTC", 
                "-07-11 23:59:59 UTC", "-07-21 23:59:59 UTC", 
                "-08-01 23:59:59 UTC", "-08-11 23:59:59 UTC", 
                "-08-21 23:59:59 UTC")



biao1 <- 0
biao2 <- 50
for (year in 2003:2022) {
  timebegin <- paste(as.character(rep(year,length(timebeginmon))), timebeginmon, sep="")
  timeend <- paste(as.character(rep(year,length(timeendmon))), timeendmon, sep="")
  for (i in 1:9) {
    indextime <- subset(indexAll, time=list(from=timebegin[i], to=timeend[i]))
    # indexa <- subset(indextime, ocean = "I")
    lonlim <- c(-180, -80)
    latlim <- c(-60, 60)
    indexa <- subset(indextime, rectangle=list(longitude=lonlim, latitude=latlim))
    profiles  <- getProfiles(indexa)
    argos <- readProfiles(profiles)
    argosClean <- applyQC(argos)
    
    name2sal <- c("id", "time", "longitude", "latitude", "sal")
    name2temp <- c("id", "time", "longitude", "latitude", "temp")
    
    df2sal <- data.frame(matrix(nrow = 0, ncol = length(name2sal)))
    df2temp <- data.frame(matrix(nrow = 0, ncol = length(name2temp)))
    
    colnames(df2sal) = name2sal
    colnames(df2temp) = name2temp
    
    
    for (j in 1:length(argosClean@data[["argos"]])) {
      
      sal <- argosClean@data[["argos"]][[j]]@data[["salinityAdjusted"]]
      if(is.null(sal) || is.null(argosClean@data[["argos"]][[j]]@data[["temperatureAdjusted"]])) {
        next
      }
#        if(is.null(argosClean@data[["argos"]][[j]]@data[["temperatureAdjusted"]]))
      dim <- dim(sal)
      sal <- as.vector(sal)
      temp <- as.vector(argosClean@data[["argos"]][[j]]@data[["temperatureAdjusted"]])
      pres <- as.vector(argosClean@data[["argos"]][[j]]@data[["pressure"]])
      longitude <- argosClean@data[["argos"]][[j]]@data[["longitude"]]
      latitude <- argosClean@data[["argos"]][[j]]@data[["latitude"]]
      if (length(longitude) < length(sal)) {
        longitude <- rep(argosClean@data[["argos"]][[j]]@data[["longitude"]], each = dim[1])
        latitude <- rep(argosClean@data[["argos"]][[j]]@data[["latitude"]], each = dim[1])
      }
      id <- rep(argosClean@data[["argos"]][[j]]@metadata[["id"]], each = dim[1])
      time <- rep(argosClean@data[["argos"]][[j]]@metadata[["time"]], each = dim[1])
      

      
      xij <- data.frame(id = id, time = time, longitude = longitude, latitude = latitude, 
                        sal = sal, temp = temp, pres = pres)
      xij <- xij[which((!is.na(xij$id)) & (!is.na(xij$time)) & (!is.na(xij$longitude)) 
                        & (!is.na(xij$latitude))) & (!is.na(xij$pres)), ]
      
      ind2sal <- which((xij$pres <= biao2) & (xij$pres >= biao1) & (!is.na(xij$sal)))
      if(length(ind2sal) > 0){
        xij2p <- aggregate(xij[ind2sal, ]$sal, by = list(xij[ind2sal, ]$id, xij[ind2sal, ]$time, 
                                                          xij[ind2sal, ]$longitude, xij[ind2sal, ]$latitude), FUN = mean)
        colnames(xij2p) <- c("id", "time", "longitude", "latitude", "sal")
        # xij2p <- data.frame(id = xij2p$id, time = xij2p$time, longitude = xij2p$longitude, latitude = xij2p$latitude, pres = xij2p$pres, sal = xij2p$sal)
        df2sal <- merge(df2sal, xij2p, all=T)
      }
      
      
      ind2temp <- which((xij$pres <= biao2) & (xij$pres >= biao1) & (!is.na(xij$temp)))    
      if(length(ind2temp) > 0){
        xij2p <- aggregate(xij[ind2temp, ]$temp, by = list(xij[ind2temp, ]$id, xij[ind2temp, ]$time, 
                                                            xij[ind2temp, ]$longitude, xij[ind2temp, ]$latitude), FUN = mean)
        colnames(xij2p) <- c("id", "time", "longitude", "latitude", "temp")
        # xij2p <- data.frame(id = xij2p$id, time = xij2p$time, longitude = xij2p$longitude, latitude = xij2p$latitude, pres = xij2p$pres, temp = xij2p$temp)
        df2temp <- merge(df2temp, xij2p, all=T)
      }
      
    }
    
    lonlim <- c(100, 180)
    latlim <- c(-60, 60)
    indexa <- subset(indextime, rectangle=list(longitude=lonlim, latitude=latlim))
    profiles  <- getProfiles(indexa)
    argos <- readProfiles(profiles)
    argosClean <- applyQC(argos)
    
    
    for (j in 1:length(argosClean@data[["argos"]])) {
      
      sal <- argosClean@data[["argos"]][[j]]@data[["salinityAdjusted"]]
      if(is.null(sal) || is.null(argosClean@data[["argos"]][[j]]@data[["temperatureAdjusted"]])) {
        next
      }
      # if(is.null(argosClean@data[["argos"]][[j]]@data[["temperatureAdjusted"]]))
      dim <- dim(sal)
      sal <- as.vector(sal)
      temp <- as.vector(argosClean@data[["argos"]][[j]]@data[["temperatureAdjusted"]])
      pres <- as.vector(argosClean@data[["argos"]][[j]]@data[["pressure"]])
      longitude <- argosClean@data[["argos"]][[j]]@data[["longitude"]]
      latitude <- argosClean@data[["argos"]][[j]]@data[["latitude"]]
      if (length(longitude) < length(sal)) {
        longitude <- rep(argosClean@data[["argos"]][[j]]@data[["longitude"]], each = dim[1])
        latitude <- rep(argosClean@data[["argos"]][[j]]@data[["latitude"]], each = dim[1])
      }
      id <- rep(argosClean@data[["argos"]][[j]]@metadata[["id"]], each = dim[1])
      time <- rep(argosClean@data[["argos"]][[j]]@metadata[["time"]], each = dim[1])
      
      
      xij <- data.frame(id = id, time = time, longitude = longitude, latitude = latitude, 
                        sal = sal, temp = temp, pres = pres)
      xij <- xij[which((!is.na(xij$id)) & (!is.na(xij$time)) & (!is.na(xij$longitude)) 
                        & (!is.na(xij$latitude))) & (!is.na(xij$pres)), ]
      
      ind2sal <- which((xij$pres <= biao2) & (xij$pres >= biao1) & (!is.na(xij$sal)))
      if(length(ind2sal) > 0){
        xij2p <- aggregate(xij[ind2sal, ]$sal, by = list(xij[ind2sal, ]$id, xij[ind2sal, ]$time, 
                                                          xij[ind2sal, ]$longitude, xij[ind2sal, ]$latitude), FUN = mean)
        colnames(xij2p) <- c("id", "time", "longitude", "latitude", "sal")
        # xij2p <- data.frame(id = xij2p$id, time = xij2p$time, longitude = xij2p$longitude, latitude = xij2p$latitude, pres = xij2p$pres, sal = xij2p$sal)
        df2sal <- merge(df2sal, xij2p, all=T)
      }
      
      
      ind2temp <- which((xij$pres <= biao2) & (xij$pres >= biao1) & (!is.na(xij$temp)))    
      if(length(ind2temp) > 0){
        xij2p <- aggregate(xij[ind2temp, ]$temp, by = list(xij[ind2temp, ]$id, xij[ind2temp, ]$time, 
                                                            xij[ind2temp, ]$longitude, xij[ind2temp, ]$latitude), FUN = mean)
        colnames(xij2p) <- c("id", "time", "longitude", "latitude", "temp")
        # xij2p <- data.frame(id = xij2p$id, time = xij2p$time, longitude = xij2p$longitude, latitude = xij2p$latitude, pres = xij2p$pres, temp = xij2p$temp)
        df2temp <- merge(df2temp, xij2p, all=T)
      }
      
    }
    
    
    data2sal <- as.matrix(df2sal[, c(3,4,5)])
    data2temp <- as.matrix(df2temp[, c(3,4,5)])
    
  save(data2sal, file=paste('RealData/Argo/data/',"year",year,"day",i,'.Rdata',sep=''))
  }
}



