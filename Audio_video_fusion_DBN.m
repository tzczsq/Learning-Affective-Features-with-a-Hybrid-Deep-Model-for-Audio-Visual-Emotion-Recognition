%%%%%%%%%%%%%%%%%% reading audio data %%%%% ¶þÎ¬Êý¾Ý
clc, close all;
clear all;
di=dir('F:\RML_wave_programe\logms_contextRGB64_64shift30\*.mat');
n=length(di);%%% n: subject number

K=n;  %%% K-fold subject_group cross-validation, if K (K=n)is the number of speaker, it is LOGO test.
for i=1:n
    Alldata1(1,i)=load(['F:\RML_wave_programe\logms_contextRGB64_64shift30\',di(i).name]);
end
clear di;

video_num=cell(n,1);
for i=1:n
video_num{i,1}=Alldata1(i).num_add;
end

label_need=cell(n,1);
for i=1:n
label_need{i,1}=Alldata1(i).label;
end

clear Alldata1;

%%%%%%%%%%%% read alex fine_tuning audio data %%%% 
di=dir('F:\RML_wave_programe\64_64_fine_tuning_AlexNet\CNN_4096D\*.mat');
n=length(di);%%% n: subject number

K=n;  %%% K-fold subject_group cross-validation, if K (K=n)is the number of speaker, it is LOGO test.
for i=1:n
    Alldata(1,i)=load(['F:\RML_wave_programe\64_64_fine_tuning_AlexNet\CNN_4096D\',di(i).name]);
end
clear di;

%%%%%%%%%%% added number 
for i=1:n
Alldata(i).num_add=video_num{i};
end
for i=1:n
Alldata(i).label=label_need{i};
end
clear video_num label_need;

%%%%%%%%%%%% read video data with  fine tuning %%%%%%%%%%%%%%%%%%%
di=dir('F:\RML_video_programe\RML_video_3DCNN_Caffe\3DCNN_feature_64_64frames\features_finetuning\*.mat');
n=length(di);%%% n: subject number

K=n;  %%% K-fold subject_group cross-validation, if K (K=n)is the number of speaker, it is LOGO test.
for i=1:n
    Alldata2(1,i)=load(['F:\RML_video_programe\RML_video_3DCNN_Caffe\3DCNN_feature_64_64frames\features_finetuning\',di(i).name]);
end
clear di;
%%% combine video and audio data %%%%%

for  j=1:K
  Audio=Alldata(j).testend_data;
  Video=Alldata2(j).testend_data;
 k1=length(Audio);
for i=1:k1
Audio{i,1}=[Audio{i,1};Video{i,1}];   
end
Alldata(j).test_new=Audio;
end
clear Audio Video;

for  j=1:K
  Audio=Alldata(j).trainend_data;
  Video=Alldata2(j).trainend_data;
 k1=length(Audio);
for i=1:k1
Audio{i,1}=[Audio{i,1};Video{i,1}];   
end
Alldata(j).train_new=Audio;
end
clear Audio Video Alldata2;

class_num=6; %%%
class_name={'anger','disgust','fear','joy','sadness','surprise'} ;%%%labels:	1-anger,2-disgust,3-fear,4-joy,5-sadness,6-surprise	
confusion_matrix_CNN=zeros(class_num,class_num); % class number:7
confusion_matrix_SVM_global=zeros(class_num,class_num);


accuracy_CNN=zeros(K,1);
accuracyVote_CNN=zeros(K,1);
accuracy_SVM_mean_global=zeros(K,1);
  for i=1:K
     
 Xtrain= Alldata(i).train_new;  %%% CNN_4096D features
Xtest=Alldata(i).test_new;
trainlabel_new=Alldata(i).trainlabel_new;
correct=Alldata(i).correct;  %%% testlabel 
      
  fprintf('Perform Kfold = %d\n', i);
   %%%%%%%%%%%%%%%% contaminate all testing data %%%%%%%%%%%%%
    testdata=[];
    w_test=[];
	 for j=1:length(Xtest)
    testdata0= Xtest{j,1};
    w1=size(testdata0,2);
     testdata=[testdata testdata0];
     w_test=[w_test w1];
     end

    testlabel=[];
    testlabel0=Alldata(i).label;
    for j3=1:length(testlabel0)
        label0=testlabel0{j3,1};
        testlabel=[testlabel;label0];
    end
    clear testdata0
   

  
   %%%%%%%%%%%%%%%% contaminate all training data %%%%%%%%%%%%% 
   
    trainlabel=[];
    w_train=[];
    traindata=[];
	 for j=1:length(Xtrain)
    traindata0= Xtrain{j,1};
    w0=size(traindata0,2);
    trainlabel0=trainlabel_new(j,1); %%% uterance label
    trainlabel1=repmat(trainlabel0,w0,1);   %%%% segment label
    %traindata=cat(2,traindata,traindata0); 
     traindata=[traindata traindata0];   %%% feature_dim*number single
     w_train=[w_train w0];
     trainlabel=[trainlabel;trainlabel1];
     end
   clear traindata0

 %%%%%  DBN learned features %%%%%%%%%%%%%  
 data = MNIST.prepareMNIST_Small('+MNIST\');% Prepare for the input data type of DBN.
 % Data value type is gaussian because the value can be consider a real
% value [-Inf +Inf]
data.valueType=ValueType.gaussian;
data.trainData=double(traindata');
data.trainLabels=trainlabel;
data.testData=double(testdata');
data.testLabels=testlabel;
 data.normalize('minmax');
data.validationData=data.testData;
data.validationLabels=data.testLabels;

dbn=DBN();
dbn.dbnType='autoEncoder';
% RBM1
rbmParams=RbmParameters(4096,ValueType.binary);  %%%% for multi-layer DBN
rbmParams.maxEpoch=100;
rbmParams.gpu=1;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);
%RBM2
% rbmParams=RbmParameters(1000,ValueType.binary);
% rbmParams.maxEpoch=50;
% rbmParams.gpu=1;
% rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
% rbmParams.performanceMethod='reconstruction';
% dbn.addRBM(rbmParams);
% % RBM3
% rbmParams=RbmParameters(2048,ValueType.binary);
% rbmParams.maxEpoch=100;
% rbmParams.gpu=1;
% rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
% rbmParams.performanceMethod='reconstruction';
% dbn.addRBM(rbmParams);

%%RBM4
rbmParams=RbmParameters(1024,ValueType.gaussian);
rbmParams.maxEpoch=100;
rbmParams.gpu=1;
rbmParams.samplingMethodType=SamplingClasses.SamplingMethodType.CD;
rbmParams.performanceMethod='reconstruction';
dbn.addRBM(rbmParams);

dbn.train(data);
% save('dbn.mat','dbn');
useGPU='no';  %%% "yes" emerges an error.
dbn.backpropagation(data,useGPU);
% save('dbn+BP.mat','dbn');

 testdata_feature=dbn.getFeature(data.testData);
 traindata_feature=dbn.getFeature(data.trainData);
   
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    num1=length(testlabel);
    num2=length(trainlabel);

test_end=testdata_feature';
testend_data=[];
b=cumsum(w_test');
for j2=1:length(b)
    if j2<2
       testend_data{j2,1}= test_end(:,1:b(j2));
    else
    testend_data{j2,1}=test_end(:,(b(j2-1)+1):b(j2));
    end
end


train_end=traindata_feature';
trainend_data=[];
a=cumsum(w_train');
for j1=1:length(a)
    if j1<2
       trainend_data{j1,1}= train_end(:,1:a(j1));

    else
    trainend_data{j1,1}=train_end(:,(a(j1-1)+1):a(j1));
    end
end


trainmax_end1=[];
for j=1:length(trainend_data)
data=double(trainend_data{j,1});
avg1=mean(data,2);
std1=std(data,0,2);
data1=deltas(data,2);
avg2=mean(data1,2);
std2=std(data1,0,2);
trainmax1=[avg1'];
% trainmax1=[avg1' std1'];
% trainmax1=[avg1' std1' avg2' std2'];
trainmax_end1=[trainmax_end1;trainmax1];
end


testmax_end1=[];
for j=1:length(testend_data)
data=double(testend_data{j,1});
avg1=mean(data,2);
std1=std(data,0,2);
data1=deltas(data,2);
avg2=mean(data1,2);
std2=std(data1,0,2);
testmax1=[avg1'];
% testmax1=[avg1' std1'];
% testmax1=[avg1' std1' avg2' std2'];
testmax_end1=[testmax_end1;testmax1];
end

 trainmax_end1=svmscale(trainmax_end1); 
 trainmax_end1(isnan(trainmax_end1))=0;
 trainmax_end1(isinf(trainmax_end1))=0; 
 testmax_end1=svmscale(testmax_end1);
 testmax_end1(isnan(testmax_end1))=0; 
 testmax_end1(isinf(testmax_end1))=0;
 
%%%%%%%%%%%%%%%% linear SVM %%%%%%%%%%%%%%%%%%
 w12= svmtrain(trainlabel_new,trainmax_end1, '-t 0 -b 1');
 [predict_label1, accuracy1, prob_estimates1] = svmpredict(correct, testmax_end1, w12,'-b 1');
accuracy_SVM_mean_global(i,1)=accuracy1(1);

 confusematrix_SVM_global=confmat1(predict_label1,correct);
 confusion_matrix_SVM_global=confusion_matrix_SVM_global+confusematrix_SVM_global;

end





