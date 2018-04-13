h%%%%%%%%%%%%%%%%%% perform CNN on context data %%%%% ¶þÎ¬Êý¾Ý
clc, close all;
clear all;
 di=dir('F:\RML_wave_programe\logms_contextRGB64_64shift30\*.mat');
n=length(di);%%% n: subject number

 K=n;  %%% 10 or less than 10 speakers databases
for i=1:n
 Alldata(1,i)=load(['F:\RML_wave_programe\logms_contextRGB64_64shift30\',di(i).name]);
end
clear di;


class_num=6; 
class_name={'anger','disgust','fear','joy','sadness','surprise'} ;%%%labels:	1-anger,2-disgust,3-fear,4-joy,5-sadness,6-surprise	
confusion_matrix_CNN=zeros(class_num,class_num); 
confusion_matrix_SVM_Average_pooling=zeros(class_num,class_num);

% rp = randperm(n); 
rp=1:n;
kappa = floor(n/K);

accuracy_CNN=zeros(K,1);
accuracyVote_CNN=zeros(K,1);
accuracy_SVM_global_Average_pooling=zeros(K,1);

 for i=1:K
  testidx = rp((i-1)*kappa + 1:i*kappa);
  trainidx = setdiff(rp(1:K*kappa), testidx); % trainidx
  Xtest = Alldata(testidx);
  Xtrain = Alldata(trainidx);
  fprintf('Perform Kfold = %d\n', i);
   %%%%%%%%%%%%%%%% contaminate all testing data %%%%%%%%%%%%%
   testlabel=[];
    testdata=[];
    w_test=[];
	 for j=1:length(Xtest)
    testdata0= Xtest(j).logms_RGB;
     testdata1=[];
    for j2=1:length(testdata0)
        testdata11=testdata0{j2,1};
          testdata11=imresize(testdata11,[227 227],'bilinear'); %%%%% alex model
        testdata1=cat(4,testdata1,testdata11);
    end
    testdata=cat(4,testdata,testdata1);
    label0=Xtest(j).label;
    label11=[];
    for j3=1:length(label0)
        label00=label0{j3,1};
        label11=[label11;label00];
    end
    testlabel=[testlabel;label11];
    w0=Xtest(j).num_add;
    w_test=[w_test, w0];
     end
    clear testdata1 
   
   %%%%%%%%%%%%%%%% contaminate all training data %%%%%%%%%%%%% 
   trainlabel=[];
    traindata=[];
    w_train=[];
	 for j=1:length(Xtrain)
    traindata0= Xtrain(j).logms_RGB;
     w0=Xtrain(j).num_add;
    w_train=[w_train, w0];
     traindata1=[];
    for j2=1:length(traindata0)
        traindata11=traindata0{j2,1};
        traindata11=imresize(traindata11,[227 227],'bilinear');  %%%  alex model
        traindata1=cat(4,traindata1,traindata11);
    end
    traindata=cat(4,traindata,traindata1);
    label0=Xtrain(j).label;
    label11=[];
    for j3=1:length(label0)
        label00=label0{j3,1};
        label11=[label11;label00];
    end
    trainlabel=[trainlabel;label11];
   
     end
   clear traindata1
   
   
   
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    num1=length(testlabel);
    num2=length(trainlabel);
 cd ('F:\RML_wave_programe');
trainOpts.batchSize = 30 ;
trainOpts.numEpochs = 40; % It is careful to change the number of Epoch, such as 10,20,30,40,50, .......
trainOpts.continue = false ;
trainOpts.gpus = [1] ;%%%[1,2] ;
trainOpts.expDir = 'RML-experiment' 
%trainOpts.expDir = subdir;
opts.whitenData=false;

    
 %%%%%%%%%%% normalize training data and testing data %%%%%
 imageMean = mean(traindata, 4);
traindata = bsxfun(@minus, traindata, imageMean) ;
std_train=std(traindata,0,4);
traindata =bsxfun(@rdivide,traindata ,std_train);
traindata(isinf(traindata))=0;
traindata(isnan(traindata))=0;


testdata= bsxfun(@minus, testdata, imageMean) ;
testdata =bsxfun(@rdivide,testdata ,std_train);
testdata(isinf(testdata))=0;
testdata(isnan(testdata))=0;

 traindata=single(traindata);
testdata=single(testdata);

[p1,p2,channels,number]=size(traindata);
if opts.whitenData
  z = reshape(traindata,[],number) ;
  W = z*z'/number ; %%% number can be deleted here.
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  traindata = reshape(z, p1, p2, channels, []) ;
end

[p1,p2,channels,number]=size(testdata);
if opts.whitenData
  z = reshape(testdata,[],number) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  testdata = reshape(z, p1, p2, channels, []) ;
end
clear W V D d2 en z;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 imdb.images.data= cat(4,traindata,testdata);  %%contaminate training data and testing data
 imdb.images.labels= cat(2,trainlabel',testlabel'); %%%% double
imdb.meta.classes=num2str(1:class_num);
%%%%%%%%%%%% train, val, test data %%%%%%%%%%%%  
   imdb.meta.sets = {'train', 'val','test'} ;
   val_train=ones(1,num2);
  val_train(1:8:end)=2;
  set=[val_train 3*ones(1,num1)];
  imdb.images.set=set; 

 net = load('F:\matconvnet\imagenet-caffe-alex.mat');% Alex  models. fc6=4096
 net.layers = net.layers(1:end-2);
 net.layers{end+1} = struct('type', 'conv', ...
 'weights', {{0.005*randn(1,1,4096,6, 'single'), zeros(1,6,'single')}}, ...  %%% class_num=6 or 7 for different databases
 'learningRate', [0.001 0.001],...
'stride', [1 1] , ...
 'pad', [0 0 0 0]) ;
 net.layers{end+1} = struct('type', 'softmaxloss') ;
 
[net,info] = cnn_train(net, imdb, @getBatch_color, trainOpts) ; %%%% getBatch_color for RGB

% % Move the CNN back to the CPU if it was trained on the GPU
if trainOpts.gpus
  net = vl_simplenn_move(net, 'cpu') ;
end
% 

 correct = testlabel(cumsum(w_test'));
 trainlabel_new=trainlabel(cumsum(w_train'));

test_end=[];
test_local=[];
predict=[];
scores_test=[];
net.layers{end}.type = 'softmax';
for k=1:num1
res_test = vl_simplenn(net, testdata(:,:,:,k)) ;
test_new1=res_test(end-2).x; %%% 4096D
p=size(test_new1,3);
scores = squeeze(gather(res_test(end).x)) ;
[bestScore, best] = max(scores) ;
scores_test=[scores_test;scores'];
predict=[predict;best];
test_new1=reshape(test_new1,[1 1*p]); 
test_end=[test_end;test_new1];

end
predictions = count_votes(scores_test,w_test');


accuracy_CNN(i,1)= sum(~(predict - testlabel))/length(testlabel);

accuracyVote_CNN(i,1) = sum(predictions==correct)/length(predictions);



% confusematrix_CNN=confmat1(predictions,correct);
% confusion_matrix_CNN=confusion_matrix_CNN+confusematrix_CNN;


testend_data=[];
testend_score=[];
testlocal_data=[];
b=cumsum(w_test');
for j2=1:length(b)
    if j2<2
       testend_data{j2,1}= test_end(1:b(j2),:)';
       testend_score{j2,1}=scores_test(1:b(j2),:)';
    else
    testend_data{j2,1}=test_end((b(j2-1)+1):b(j2),:)';
    testend_score{j2,1}=scores_test((b(j2-1)+1):b(j2),:)';
    end
end


train_end=[];
scores_train=[];
train_local=[];
for k=1:num2
res_train = vl_simplenn(net,traindata(:,:,:,k)) ;
train_new1=res_train(end-2).x; %%% 4096D 
p=size(train_new1,3);
scores = squeeze(gather(res_train(end).x)) ;
[bestScore, best] = max(scores) ;
scores_train=[scores_train;scores'];
train_new1=reshape(train_new1,[1 1*p]);  
train_end=[train_end;train_new1];
end

trainend_data=[];
trainend_score=[];
trainlocal_data=[];
a=cumsum(w_train');
for j1=1:length(a)
    if j1<2
       trainend_data{j1,1}= train_end(1:a(j1),:)';
       trainend_score{j1,1}=scores_train(1:a(j1),:)';
      
    else
    trainend_data{j1,1}=train_end((a(j1-1)+1):a(j1),:)';
     trainend_score{j1,1}=scores_train((a(j1-1)+1):a(j1),:)';
     
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
%trainmax1=[avg1' std1']; %% add the std information
%trainmax1=[avg1' std1' avg2' std2']; %% add the delta inforamtion
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
%testmax1=[avg1' std1'];
%testmax1=[avg1' std1' avg2' std2'];
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
accuracy_SVM_global_Average_pooling(i,1)=accuracy1(1);

save(['netmodel' num2str(i)], 'net');
save(['LOSO_CNNdata' num2str(i)], 'trainend_data','testend_data','correct','trainlabel_new');
save(['LOSO_CNNscoredata' num2str(i)], 'trainend_score','testend_score','correct','trainlabel_new');

end
