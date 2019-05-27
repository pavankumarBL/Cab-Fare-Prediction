rm(list=ls())
setwd('C:\\Users\\pavankumar.bl\\Documents\\datascience\\Edwisor\\Project_2\\R')
train_cab=read.csv("C:\\Users\\pavankumar.bl\\Documents\\datascience\\Edwisor\\Project_2\\R\\train_cab.csv",header=TRUE,strip.white = TRUE,stringsAsFactors = FALSE)
#Load the Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','dplyr','lubridate',
      'tidyverse','geosphere','tictoc')

lapply(x,require,character.only=TRUE)
rm(x)
#EDA
#replacing all the 0's in Data as NA
train_cab[train_cab == 0] <- NA
#checkin total Missing values
apply(train_cab, 2, function(x){sum(is.na(x))})

#checking the Structure of the Training data
str(train_cab)
#Converting Fare amount to Floating i.e. Numeric
train_cab$fare_amount=as.numeric(as.character(train_cab$fare_amount))
#extracting the date time year month from date column and store the each variable in Training dataframe
train_cab <- mutate(train_cab,
                    pickup_datetime = ymd_hms(`pickup_datetime`),
                    month = as.integer(month(pickup_datetime)),
                    year = as.integer(year(pickup_datetime)),
                    dayOfWeek = as.integer(wday(pickup_datetime)),
                    hour = hour(pickup_datetime),
                    hour = as.integer(hour(pickup_datetime))
)


#======================Missing Value=====================#
Missing_val=data.frame(apply(train_cab,2,function(x){sum(is.na(x))}))
Missing_val$Columns=row.names(Missing_val)
row.names(Missing_val)=NULL
names(Missing_val)[1]= "Missing_values"
#finding missing value percentage for all variable in data set
Missing_val$Missing_percentage=(Missing_val$Missing_values/nrow(train_cab))*100
#descending order
Missing_val=Missing_val[order(-Missing_val$Missing_percentage),]
#plotting the graph
ggplot(data = Missing_val[1:15,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
  geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
  ggtitle("Missing data percentage (Train)") + theme_bw()


#Outlier Detection and Removal
train_cab = train_cab %>% 
  filter(pickup_latitude<=90 , pickup_latitude>=-90 ,
         dropoff_latitude<=90 , dropoff_latitude>=-90  ,
         pickup_longitude<=180 , pickup_longitude>=-180 ,
         dropoff_longitude<=180 , dropoff_longitude>=-180)

train_cab = train_cab %>%
  filter(fare_amount>=1,fare_amount<=200)

train_cab = train_cab %>%
  filter(passenger_count>=0,passenger_count<=8)

train_cab=na.omit(train_cab)


apply(train_cab,2,function(x){sum(is.na(x))})
summary(train_cab)
#Feature Engineering
train_cab = train_cab %>% 
  mutate(distance.in.KM = by(train_cab, 1:nrow(train_cab), function(row) { 
    distHaversine(c(row$pickup_longitude, row$pickup_latitude), c(row$dropoff_longitude,row$dropoff_latitude))/1000}))
summary(train_cab$distance.in.KM)
str(train_cab)
#Removing zero distance value
train_cab = train_cab %>%
  filter(distance.in.KM>0,distance.in.KM<=100)
summary(train_cab$distance.in.KM)

#PLoting the graph for distance and Fare
gplot <- ggplot(data=train_cab, aes(x=train_cab$distance.in.KM, y=train_cab$fare_amount)) + geom_point()+ geom_line()+ 
  ggtitle("Distance and Fare Plot") +
  xlab("Distance in KM ") + 
  ylab("Fare")
gplot
#PLoting the graph for Hour of the Day and Fare
ggplot(train_cab,aes(x = hour, y=fare_amount))+
  geom_line()+
  labs(x= "hour of the day")+
  scale_x_discrete(limits = c(0:23))+
  scale_y_continuous(limits=c(0,180))
#From the above graph we can see that the timeing is not affecting too much. Maximin dots are below 100. 
#PLoting the graph for passenger count and Fare
gplot_p <- ggplot(data=train_cab, aes(x=passenger_count, y=fare_amount)) + geom_point()+ geom_line()+ 
  ggtitle("Time and Fare Plot") +
  xlab("Passenger Count ") + 
  ylab("Fare")
gplot_p
# From the Graph it seems passenger count is not affecting the fare.
#Frequency of 1 passenger is high.
hist(train_cab$passenger_count,xlab = "Passenger count ", main=paste("Hist of Passenger count"),col = "Yellow")
#Plotting the graph for passenger Day count and Fare
gplot_d <- ggplot(data=train_cab, aes(x=dayOfWeek, y=fare_amount)) + geom_point()+ geom_line()+ 
  ggtitle("Day count and Fare Plot") +
  xlab("Day Count ") + 
  ylab("Fare")
gplot_d
#=================Building the Model=================#
train_cab_final <- subset(train_cab, select = c(1,7:12))
view(train_cab_final)
str(train_cab_final)

linerModel <- lm(fare_amount~ distance.in.KM, data = train_cab_final)
summary(linerModel)
predict_fare_simple<-predict(linerModel, newdata = train_cab_final)
view(train_cab_final)
view(predict_fare_simple)

#====================Multiple Linear Regression Model=============#
#selecting the necessary variable and keeping it for model
train_cab_final <- subset(train_cab, select = c(1,7:12))
view(train_cab_final)
str(train_cab_final)

#fit the Model with fare amoun and independant variables.
linerModel <- lm(fare_amount~., data = train_cab_final)
summary(linerModel)
#predicting the Fare amount for Test data
predict_fare_lm<-predict(linerModel, newdata = train_cab_final)
view(train_cab_final)
view(predict_fare_lm)
#=============Random Forest========================#
tic()
fitRF <- randomForest(formula = fare_amount ~ ., data = train_cab_final, ntree = 200)
toc()
# Checking model by predicting on out of sample data
predictRF <- predict(fitRF, train_cab_final)
print(fitRF)
# Using root mean squared as error function
rmseRF <- sqrt(sum((predictRF - train_cab_final$fare_amount)^2) / nrow(train_cab_final))
print(rmseRF/mean(train_cab_final$fare_amount))
view(predictRF)
#===========Predicting for Test Data========================#
test_cab=read.csv("test_cab.csv",header = TRUE,stringsAsFactors = FALSE)
apply(test_cab, 2, function(x){sum(is.na(x))})
str(test_cab)
test_cab <- mutate(test_cab,
                    pickup_datetime = ymd_hms(`pickup_datetime`),
                    month = as.integer(month(pickup_datetime)),
                    year = as.integer(year(pickup_datetime)),
                    dayOfWeek = as.integer(wday(pickup_datetime)),
                    hour = hour(pickup_datetime),
                    hour = as.integer(hour(pickup_datetime))
)
summary(test_cab)
test_cab = test_cab %>% 
  mutate(distance.in.KM = by(test_cab, 1:nrow(test_cab), function(row) { 
    distHaversine(c(row$pickup_longitude, row$pickup_latitude), c(row$dropoff_longitude,row$dropoff_latitude))/1000}))
summary(test_cab$distance.in.KM)
str(test_cab)
view(test_cab)
test_cab_final <- subset(test_cab, select = c(6:11))
view(test_cab_final)
str(test_cab_final)
predictRF_test <- predict(fitRF, test_cab_final)

#================Comparing the result==============#
train_cab_final$predict_fare_lm=predict_fare_lm
train_cab_final$predictRF=predictRF
view(test_cab_final)
test_cab_final$fare_amount=predictRF_test
#PLoting the graph for distance and Fare
gplot <- ggplot(data=test_cab_final, aes(x=distance.in.KM, y=fare_amount)) + geom_point()+ geom_line()+ 
  ggtitle("Distance and Fare Plot") +
  xlab("Distance in KM ") + 
  ylab("Fare")
gplot
#=============END of the Script===========#
