clear all;
for k=1:8; % %%
 
subjectname=['s',num2str(k)];
subdir=strcat('G:\RML_wavdatabase','\',subjectname);
filename=dir(fullfile(subdir,'*.wav'));
 dirIndex = [filename.isdir];  %# Find the index for directories
 fileList = {filename(~dirIndex).name}';  %'# Get a list of the files
n= numel(fileList);
label=[];
label_context=[];
 label_add_context=[];
  label_add=[];
 logmsdata=[];
 gbfbdata=[];
 sgbfbdata=[];
logms_context=[];
gbfb_context=[];
sgbfb_context=[];
num_context_add=[];
num_add=[];
 mean_logs=[];
 
 logms_RGB=[];
 
for j=1:n
     file_name = char(fileList(j));
     fprintf('Processing image: %d/%d\n', i, n);
    
     waveFile = strcat(subdir,'\',file_name);
if findstr(file_name,'angry');  
    label1='1';
    label1 = str2double(label1);
    elseif findstr(file_name,'happy');  				
    label1='2';
    label1 = str2double(label1);
    elseif findstr(file_name,'sad');  				
    label1='3';
    label1 = str2double(label1);
    elseif findstr(file_name,'fear');  				
    label1='4';
    label1 = str2double(label1);
    elseif findstr(file_name,'disgust');  				
    label1='5';
    label1 = str2double(label1);
     else				
    label1='6';
    label1 = str2double(label1);
end

num_bands=64;
context_size=64; %%% context window size
shift=30;  %%% 

win_shift=10;
 win_length = 25;
freq_range=[20, 8000];
 [signal, fs]=audioread(waveFile);  
 signal=signal(:,1);% change 2 channel into 1 channel(1-left channel;2-right channel)

logms= log_mel_spectrogram(signal, fs, win_shift, win_length, freq_range, num_bands); 
 logms_context0=context_window(logms,context_size,num_bands,shift);
 num_frame=size(logms_context0,1);
  logms_RGB2=zeros(num_bands,context_size,3,num_frame);
 logms_RGB1=zeros(num_bands,context_size,3);
 for kk=1:num_frame
     log2=logms_context0{kk,1};
      del = deltas(log2); %%% ;
     ddel = deltas(del);%%
   logms_RGB1(:,:,1)=log2;
   logms_RGB1(:,:,2)=del;
   logms_RGB1(:,:,3)=ddel;
   logms_RGB2(:,:,:,kk)=logms_RGB1;
 end
logms_RGB{j,1}=logms_RGB2;
num_add=[num_add,num_frame];
amount=[file_name,'- ',num2str(num_frame)];
label_add=[label_add,amount];

logs_add=repmat(label1,num_frame,1);
  label{j,1}=logs_add;
end
 
  cd('G:\RML_wave_programe\logms_contextRGB64_64shift30');
   save(strcat('logms_RGB', num2str(k)),'logms_RGB','label','label_add','num_add');
end
 
 
 
