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
# �����ʹ�Sine graph�� ������� �ϰ� random���� 100���� selection�ϴ���
# �� 20�� ������ Uniform������ Noise�� �ְ� ���� Class activation map��
# �� ������ �����Ϳ��� �� �۵��ϴ����� ������ �Ѵ�.

t=seq(0,1000,0.1)
Sine_Graph=sin(t)
Another_Graph=sin(t)
t2=seq(0,9999,100)
for (i in t2){Another_Graph[i:(i+20)]<-rnorm(1)}#end i 




#�ð�ȭ �ؼ� �ѹ� ������ ����.
Data_Visualization<-function()
{
  par(mfrow=c(2,1))
  Index<-sample(1:(length(Sine_Graph)-100),1)
  plot(Sine_Graph[Index:(Index+99)],type='o')
  plot(Another_Graph[Index:(Index+99)],type='o')
}#end function

Data_Visualization()

# �׷� ���� rawdata�� batch 32�� �������� �����ؼ�
# ���� 16,000���� ���� binary classification dataset�� ����� ������ ����.

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

# �� ���� �����͵� ����� ������ Numpy�� ���� tensorflow�� ���� �Ͽ� ����.
library(RcppCNPy)
npySave("Toy_Dataset_Sine.npy", Input_Rawdata)














