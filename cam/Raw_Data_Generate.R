# '--------------------------------------------------------------------------
#  Junhong Kim
#  Ph.D. Student in Industrial Management Engineering
#  Korea University, Seoul, Republic of Korea
#  Mobile Phone +82 10 3099 3004
#  E-mail    junhongkim@korea.ac.kr
#  Data Science and Business Analytics Lab
#  Lab Homepage http://dsba.korea.ac.kr
# --------------------------------------------------------------------------'




# 'SET_Working_Directory!'
Working_Directory = getwd()
setwd(Working_Directory)
# 데이터는Sine graph를 기반으로 하고 random으로 100개를 selection하더라도
# 약 20개 정도가 Uniform분포의 Noise로 넣고 과연 Class activation map이
# 이 간단한 데이터에서 잘 작동하는지를 보도록 한다.

t=seq(0,1000,0.1)
Sine_Graph=sin(t)
Another_Graph=sin(t)
t2=seq(0,9999,100)
for (i in t2){Another_Graph[i:(i+20)]<-rnorm(1)}#end i 




#시각화 해서 한번 보도록 하자.
Data_Visualization<-function()
{
  par(mfrow=c(2,1))
  Index<-sample(1:(length(Sine_Graph)-100),1)
  plot(Sine_Graph[Index:(Index+99)],type='o')
  plot(Another_Graph[Index:(Index+99)],type='o')
}#end function

Data_Visualization()

# 그럼 이제 rawdata를 batch 32를 잡을것을 생각해서
# 각각 16,000개씩 만들어서 binary classification dataset을 만들어 보도록 하자.

Sampling_Function<-function(Data)
{
  Index<-sample(1:(length(Data)-100),1)
  return(Data[Index:(Index+99)])
} #end Sample_Function 

Class_One_List<-list()
Class_Two_List<-list()

for (i in 1:16000)
{
  Class_One_List[[i]]<-Sampling_Function(Sine_Graph)
  Class_Two_List[[i]]<-Sampling_Function(Another_Graph)
  print(i)
}#end i 

Class_One<-do.call(rbind,Class_One_List)
Class_Two<-do.call(rbind,Class_Two_List)


Input_Rawdata<-rbind(Class_One,Class_Two)
Target<-matrix(c(rep(c(1,0),16000),rep(c(0,1),16000)),ncol=2,byrow=T)


Input_Rawdata <- data.frame(cbind(Input_Rawdata,Target))

colnames(Input_Rawdata)[1:100]<-paste0("T",1:100)
colnames(Input_Rawdata)[101:102]<-c("Target1","Target2")
colnames(Input_Rawdata)
Input_Rawdata<-as.matrix(Input_Rawdata)

# 자 이제 데이터도 만들어 졌으니 Numpy로 빼고 tensorflow로 진행 하여 보자.
library(RcppCNPy)
npySave("Toy_Dataset_Sine.npy", Input_Rawdata)















