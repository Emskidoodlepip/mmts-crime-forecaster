#import libraries
library(data.table)
library(magrittr)
library(plyr)
library(dplyr)
library(sf)
library(geojsonsf)
library(readxl)
library(lubridate)
library(imputeTS)
library(abind)
library(tidyr)
library(tseries)



## preprocessing
# read in from CSVs
crimes <- list.files("/Users/emily/Documents/Work/Dissertation/Data/Crime", full.names = T, recursive = T, pattern = "*west-yorkshire-street.csv") %>% ldply(read.csv, header=T)

# ditch unnecessary columns
crimes <- crimes %>% select(Crime.ID, Month, Longitude, Latitude)

# remove records with any null information as it's all needed
crimes <- na.omit(crimes)

# remove duplicate IDs
crimes <- crimes %>% distinct(Crime.ID, .keep_all = T)



## spatial processing
# spatial prep
sf_use_s2(F)

# combine coords
crimes <- st_as_sf(x = crimes, coords = c("Longitude", "Latitude")) %>% st_set_crs(value = 4326)

# import 2021 LSOA boundaries as geoJSON
lsoas <- st_read("/Users/emily/Documents/Work/Dissertation/Data/Boundaries/LSOA-2021-EW-BFE.geojson")

# infer LSOA information from boundaries, first omitting external unmatched records
crimes <- crimes[lengths(st_intersects(crimes$geometry, lsoas$geometry, prepared = T)) != 0,]
crimes$FID <- unlist(st_intersects(crimes$geometry, lsoas$geometry, prepared = T))
crimes <- crimes %>% full_join(st_drop_geometry(lsoas), by="FID")

# final cleanup before transformation!
crimes <- na.omit(st_drop_geometry(crimes) %>% select(Month, LSOA21CD))



## final transformation
# count crimes per month/location
crimes <- crimes %>% group_by(LSOA21CD, Month) %>% summarize(totalCrimes=n())

# finally, transpose to achieve wide format
crimes <- crimes %>% dcast(LSOA21CD ~ Month, value.var="totalCrimes", fill=0)

# these'll be handy later for minimising unnecessary processing
monthDomain <- colnames(crimes)[-1]
lsoaDomain <- crimes$LSOA21CD



# load mean house price data
housePrices <- read_xls("/Users/emily/Documents/Work/Dissertation/Data/Mean House Price/HPSSA47.xls", sheet="Data")

# remove unnecessary rows and columns
housePrices <- housePrices[-c(1:2, 4:68, 113)][-c(1:4),]

# rename columns by date and clean up
housePrices <- housePrices[-1,] %>% setNames(c("LSOA21CD",lapply(as.vector(head(housePrices, 1)[-1]), function(a){a %>% substring(13) %>% my() %>% format("%Y-%m")})))

# add missing columns and sort
missingHPCols <- monthDomain[-pmatch(colnames(housePrices)[-1], monthDomain)]
housePrices <- housePrices %>% cbind(setNames(lapply(missingHPCols, function(a) a=NA), missingHPCols))
housePrices <- housePrices[, c("LSOA21CD", sort(colnames(housePrices[-1])))]

# impute missing values by interpolation
housePrices[housePrices == ":"] <- NA
housePrices[-1] <- t(t(housePrices[-1] %>% sapply(as.numeric)) %>% na_ma(k=1, weighting="linear"))

# clean up
housePrices <- na.omit(housePrices)
rm(missingHPCols)

# remove unmatched records
housePrices <- housePrices[housePrices$LSOA21CD %in% crimes$LSOA21CD,]
crimes <- crimes[crimes$LSOA21CD %in% housePrices$LSOA21CD,]



# create new multivariate time series tensor
# convert dataframes to matrices
rownames(crimes) <- crimes$LSOA21CD
crimes <- as.matrix(crimes[-1])
rownames(housePrices) <- housePrices$LSOA21CD
housePrices <- as.matrix(housePrices[-1])

# define standardisation for matrices
standardise <- function(a) {return( (a-mean(a))/sd(a) )}

# back up crimes and housePrices so we can keep unstandardised data handy
tempCrimes <- crimes
tempHousePrices <- housePrices

# standardise...
crimes <- standardise(crimes)
housePrices <- standardise(housePrices)

# combine!
mts <- sapply(mget(c("crimes", "housePrices")), identity, simplify="array")

# restore backups and clean up
crimes <- tempCrimes
housePrices <- tempHousePrices
rm(tempCrimes)
rm(tempHousePrices)



# age/sex processing
# load in age/sex data
ageSexPop <- list.files("/Users/emily/Documents/Work/Dissertation/Data/Age & Sex", full.names = T, pattern = "*.xls*") %>% lapply(read_excel, sheet = 4)
ageSexFem <- list.files("/Users/emily/Documents/Work/Dissertation/Data/Age & Sex", full.names = T, pattern = "*.xls*") %>% lapply(read_excel, sheet = 6)

# match together population data from each file
ageSexMonths <- c("2012-06", "2013-06", "2014-06", "2015-06", "2016-06", "2017-06", "2018-06", "2019-06", "2020-06")
population <- data.frame(ageSexPop[[1]]$Contents[-c(1:4)])
colnames(population) <- c("LSOA21CD")
  for (y in c(1:length(ageSexMonths))){
  curYear <- data.frame(cbind(ageSexPop[[y]][[1]][-c(1:4)], ageSexPop[[y]][[3]][-c(1:4)]))
  colnames(curYear) <- c("LSOA21CD", ageSexMonths[[y]])
  population <- population %>% full_join(curYear, by="LSOA21CD")
}

# clean up population data a bit
population[-1] <- population[-1] %>% sapply(as.numeric)
rownames(population) <- population[[1]]
population <- population[-1]

# likewise, match together female population data from each file
femProportion <- data.frame(ageSexFem[[1]]$Contents[-c(1:4)])
colnames(femProportion) <- c("LSOA21CD")
for (y in c(1:length(ageSexMonths))){
  curYear <- data.frame(cbind(ageSexFem[[y]][[1]][-c(1:4)], ageSexFem[[y]][[3]][-c(1:4)]))
  colnames(curYear) <- c("LSOA21CD", ageSexMonths[[y]])
  femProportion <- femProportion %>% full_join(curYear, by="LSOA21CD")
}

# and clean up female population data a bit
femProportion[-1] <- femProportion[-1] %>% sapply(as.numeric)
rownames(femProportion) <- femProportion[[1]]
femProportion <- femProportion[-1]
rm(ageSexFem)

# finally, do the same for youth population data
ythProportion <- data.frame(ageSexPop[[1]]$Contents[-c(1:4)])
colnames(ythProportion) <- c("LSOA21CD")
for (y in c(1:length(ageSexMonths))){
  curYear <- data.frame(cbind(ageSexPop[[y]][[1]][-c(1:4)], ageSexPop[[y]][[5]][-c(1:4)]))
  colnames(curYear) <- c("LSOA21CD", ageSexMonths[[y]])
  ythProportion <- ythProportion %>% full_join(curYear, by="LSOA21CD")
}
rm(curYear)
rm(y)

# last of the cleanups, this time for youth population...
ythProportion[-1] <- ythProportion[-1] %>% sapply(as.numeric)
rownames(ythProportion) <- ythProportion[[1]]
ythProportion <- ythProportion[-1]
rm(ageSexPop)

# turn absolute female population into proportion of overall
femProportion <- femProportion/population

# and absolute youth population into proportion of overall
ythProportion <- ythProportion/population

# remove unnecessary rows...
population <- population[lsoaDomain,]
femProportion <- femProportion[lsoaDomain,]
ythProportion <- ythProportion[lsoaDomain,]

# ...and add missing cols
missingAgeSexCols <- monthDomain[-pmatch(ageSexMonths, monthDomain)]
population <- population %>% cbind(setNames(lapply(missingAgeSexCols, function(a) a=NA), missingAgeSexCols))
population <- population[, sort(colnames(population))]
femProportion <- femProportion %>% cbind(setNames(lapply(missingAgeSexCols, function(a) a=NA), missingAgeSexCols))
femProportion <- femProportion[, sort(colnames(femProportion))]
ythProportion <- ythProportion %>% cbind(setNames(lapply(missingAgeSexCols, function(a) a=NA), missingAgeSexCols))
ythProportion <- ythProportion[, sort(colnames(ythProportion))]
rm(ageSexMonths)
rm(missingAgeSexCols)

# impute by interpolation
population <- t(t(population) %>% na_ma(k=1, weighting="linear"))
femProportion <- t(t(femProportion) %>% na_ma(k=1, weighting="linear"))
ythProportion <- t(t(ythProportion) %>% na_ma(k=1, weighting="linear"))

# match LSOAs to those in mts
population <- population[dimnames(mts)[[1]],]
femProportion <- femProportion[dimnames(mts)[[1]],]
ythProportion <- ythProportion[dimnames(mts)[[1]],]

# and add standardised forms to tensor
mts <- abind(mts, standardise(population), standardise(femProportion), standardise(ythProportion))
dimnames(mts)[[3]][-c(1:2)] = c("population", "femProportion", "ythProportion")



# let's now process number of pubs per lsoa
# import CSVs, storing month data
pubs <- list.files("/Users/emily/Documents/Work/Dissertation/Data/Pubs", full.names = T, pattern = "*.csv")
names(pubs) <- paste(pubs)
pubs <- ldply(pubs, read.csv, header=T)
pubs$.id <- pubs$.id %>% substr(62,68)

# remove unnecessary columns
pubs <- pubs %>% select(.id, postcode, longitude, latitude)

# clean placeholder co-ordinates
pubs$longitude[pubs$longitude == "\\N"] <- NA
pubs$latitude[pubs$latitude == "\\N"] <- NA

# enforce numeric co-ordinates
pubs$longitude <- as.numeric(pubs$longitude)
pubs$latitude <- as.numeric(pubs$latitude)

# look up co-ordinates from 2016 postcodes for those missing them
postcodes <- ldply("/Users/emily/Documents/Work/Dissertation/Data/Boundaries/Postcode Lookup 2023-03.csv", read.csv, header=T)
postcodes <- postcodes %>% select(Postcode.3, Longitude, Latitude)
colnames(postcodes) <- c("postcode", "longitude", "latitude")
pubsLost <- pubs[is.na(pubs$longitude),] %>% inner_join(postcodes, by="postcode") %>% select(.id, postcode, longitude.y, latitude.y)
pubs <- pubs[!is.na(pubs$longitude),]
colnames(pubsLost) <- c(".id", "postcode", "longitude", "latitude")
pubs <- pubs %>% rbind(pubsLost)

# clean up
pubs <- pubs %>% select(.id, longitude, latitude)
colnames(pubs) <- c("Month", "Longitude", "Latitude")
rm(pubsLost)
rm(postcodes)

# more spatial processing, but this time check for pubs not in any LSOA as there are many, e.g. Scotland/N.I.
pubs <- st_as_sf(x = pubs, coords = c("Longitude", "Latitude")) %>% st_set_crs(value = 4326)
pubs <- pubs[lengths(st_intersects(pubs$geometry, lsoas$geometry, prepared = T)) != 0,]
pubs$FID <- unlist(st_intersects(pubs$geometry, lsoas$geometry, prepared = T))
pubs <- pubs %>% full_join(st_drop_geometry(lsoas), by="FID") %>% select(Month, LSOA21CD)
pubs <- st_drop_geometry(pubs)

# remove any LSOAs outside mts domain
pubs <- pubs[dimnames(mts)[[1]] %in% pubs$LSOA21CD,]
pubs <- pubs[pubs$LSOA21CD %in% dimnames(mts)[[1]],]

# count pubs per LSOA
pubs <- data.frame(pubs %>% group_by(LSOA21CD, Month) %>% summarize(totalPubs=n()))

# account for publess LSOAs by assuming that this will always be the case
pubs$totalPubs[is.na(pubs$Month)] <- 0
pubs$Month[is.na(pubs$Month)] <- "2016-09"

# convert from long to wide
pubs <- pubs %>% spread(Month, totalPubs)

# replace with 0 instances of no pubs causing NAs by not being present in months' lists
pubs$"2016-09"[is.na(pubs$"2016-09")] <- 0
pubs$"2018-08"[is.na(pubs$"2018-08")] <- 0
pubs$"2020-07"[is.na(pubs$"2020-07")] <- 0
pubs$"2022-02"[is.na(pubs$"2022-02")] <- 0

# another clean
rownames(pubs) <- pubs$LSOA21CD
pubs <- pubs[-1]

# add missing columns and sort
missingPubCols <- monthDomain[-pmatch(colnames(pubs), monthDomain)]
pubs <- pubs %>% cbind(setNames(lapply(missingPubCols, function(a) a=NA), missingPubCols))
pubs <- pubs[, sort(colnames(pubs))]
rm(missingPubCols)

# interpolate
pubs <- t(t(pubs) %>% na_ma(k=1, weighting="linear"))

# standardise and add to tensor
pubs <- as.matrix(pubs)
mts <- abind(mts, standardise(pubs))
dimnames(mts)[[3]][-c(1:5)] = c("pubs")



# preparation for learning
# aggregate dickey-fuller, kdss, and ljung-box tests on mean values for unit root and portmanteau metrics
mtsOld <- mts
genTests <- function(toTest = dimnames(mtsOld)[[3]], differences = 0) {
  tsTests <- data.frame(toTest)
  colnames(tsTests) <- c("Time Series")
  rownames(tsTests) <- rownames(toTest)
  tsTests$"ADF: Dickey-Fuller" <- 0
  tsTests$"ADF: Lag order" <- 0
  tsTests$"ADF: p" <- 0
  tsTests$"KPSS: KPSS level" <- 0
  tsTests$"KPSS: Truncation lag parameter" <- 0
  tsTests$"KPSS: p" <- 0
  tsTests$"LB: X-squared" <- 0
  tsTests$"LB: df" <- 0
  tsTests$"LB: p" <- 0
  for (slice in tsTests$"Time Series") {
    meanLSOA <- 1
    for (month in dimnames(mts)[[2]]) {
      print(month)
      meanLSOA <- c(meanLSOA, mean(mts[,month,slice]))
    }
    print(meanLSOA)
    meanLSOA <- ts(meanLSOA[-1], frequency = 12, end = c(2023, 2))
    if (differences > 0) { meanLSOA <- diff(meanLSOA, differences = differences) }
    # plot.ts(meanLSOA, main=slice)
    suppressWarnings({
      tsTests[slice, "ADF: Dickey-Fuller"] <- adf.test(meanLSOA)["statistic"]
      tsTests[slice, "ADF: Lag order"] <- adf.test(meanLSOA)["parameter"]
      tsTests[slice, "ADF: p"] <- adf.test(meanLSOA)["p.value"]
      tsTests[slice, "KPSS: KPSS level"] <- kpss.test(meanLSOA)["statistic"]
      tsTests[slice, "KPSS: Truncation lag parameter"] <- kpss.test(meanLSOA)["parameter"]
      tsTests[slice, "KPSS: p"] <- kpss.test(meanLSOA)["p.value"]
      tsTests[slice, "LB: X-squared"] <- Box.test(meanLSOA, type="Ljung-Box")["statistic"]
      tsTests[slice, "LB: df"] <- Box.test(meanLSOA, type="Ljung-Box")["parameter"]
      tsTests[slice, "LB: p"] <- Box.test(meanLSOA, type="Ljung-Box")["p.value"]
    })
  }
  tsTests <- tsTests[-1]
  rm(slice)
  rm(month)
  rm(meanLSOA)
  return(tsTests[ -c(1:length(toTest)), ])
}

# apply tests on all variables, peeking ahead w/r/t differencing, to identify action to achieve stationarity
View(genTests())
View(genTests(differences = 1))
View(genTests(differences = 2))

# crimes        0: a/r. none. unit root so difference.   1: r/a. adf good, kpss good. stationary.
# housePrices   0: a/r. none. unit root so difference.   1: a/r. none. unit root so difference.                                    2: r/a. adf good, kpss good. stationary.
# population    0: a/r. none. unit root so difference.   1: r/r. adf good, kpss bad. no unit root, dif. sta./stochastic. diff.     2: r/a. adf good, kpss good. stationary.
# femProportion 0: a/r. none. unit root so difference.   1: a/a. adf bad, kpss good. unit root, tre. sta./deterministic. detrend.
# ythProportion 0: a/r. none. unit root so difference.   1: r/r. adf good, kpss bad. no unit root, dif. sta./stochastic. diff.     2: r/a. adf good, kpss good. stationary.
# pubs          0: a/r. none. unit root so difference.   1: a/a. adf bad, kpss good. unit root, tre. sta./deterministic. detrend.

# difference every lsoa in a slice
sliceDiff <- function(slice, differences = 1) {
  newSlice <- slice[,-c(1:differences)]
  for (lsoa in rownames(slice)) { newSlice[lsoa,] <- diff(ts(slice[lsoa,], frequency = 12, end = c(2023, 2)), differences = differences) }
  return(newSlice)
}

# identify and remove trends after differencing in deterministic trend stationary matrices for detrending where differencing alone will not suffice
sliceDetrend <- function(slice, differences = 0) {
  newSlice <- slice
  if (differences > 0) { newSlice <- newSlice[,-c(1:differences)] }
  for (lsoa in rownames(slice)) {
    curSeries <- ts(slice[lsoa,], frequency = 12, end = c(2023, 2))
    if (differences > 0) { curSeries <- diff(curSeries, differences = differences) }
    trend <- lm(curSeries~c( 1 : (dim(mtsOld)[2] - differences) ))
    newSlice[lsoa,] <- curSeries - residuals(trend)
  }
  return(newSlice)
}

# transform for stationarity.
mts <- mts[,-c(1:2),]
mts[,,"crimes"] <- sliceDiff(mtsOld[,,"crimes"])[,-1]
mts[,,"housePrices"] <- sliceDiff(mtsOld[,,"housePrices"], differences = 2)
mts[,,"population"] <- sliceDiff(mtsOld[,,"population"], differences = 2)
mts[,,"femProportion"] <- sliceDetrend(mtsOld[,,"femProportion"], differences = 1)[,-1]
mts[,,"ythProportion"] <- sliceDiff(mtsOld[,,"ythProportion"], differences = 2)
mts[,,"pubs"] <- sliceDetrend(mtsOld[,,"pubs"], differences = 1)[,-1]

# export data by LSOA as CSV
dimnames(mts)[[2]] <- dimnames(mts)[[2]] %>% lapply(ym)
for (lsoa in dimnames(mts)[[1]]) { write.csv(mts[lsoa,,], paste0(paste0("/Users/emily/Documents/Work/Dissertation/Data/Out/", lsoa), ".csv")) }
