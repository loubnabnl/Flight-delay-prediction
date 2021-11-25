## Flight delay prediction using Gradient Boosting Trees

In this work, the goal is to predict flight delay given information
about flights in 7 delay bins,
where 0 means no delay and 7 is a delay superior to 2 hours. We have a training dataset 
with flights information as well as the expected and actual arrival dates, and we want to predict the delay for
a test set (without knowing the actual arrival date ofc), we also had acess to a dataset
with information about the distance and countries of origin and destination.. but the use of external datasets was not allowed.""" 

We applied different preprocessing techniques on the data to get a rich set of features, and we used a Gradient Boosting Trees to predict
the actual arrival time of flights, than we applied an optimized thresholding approach to place the delays in the desired bins.