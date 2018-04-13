# audio-visual-emotion-recognition-using-a-hybrid-deep-model

This is just an example code for audio-visual emotion recognition via a hybrid deep model (CNN for audio feature extraction, 3D-CNN for visual feature extraction, DBN for audio-visual fusion) published on TCSVT-2017. Here is to show the main idea related to TCSVT-2017. I do not upload all the related files since I realize them under two different systems. For instance, 3D-CNN is performed with Caffe on Linux, whereas other deep models are implemented on Windows. These related tools are given below. Note that carefully adjust the parameters of deep models. 

Related datasets and tools can be found as follows:
RML audio-visual emotional database: https://github.com/emotion-database/RML_Emotion_Database

Log Mel_spectorgram extraction: https://github.com/m-r-s/reference-feature-extraction

MatConvnet:  http://www.vlfeat.org/matconvnet/

3D-CNN(Caffe): http://caffe.berkeleyvision.org/

DBN: http://ceit.aut.ac.ir/ keyvanrad/

SVM: https://www.csie.ntu.edu.tw/cjlin/libsvm/


Please see the details and cite our paper:

Zhang Shiqing, Zhang Shiliang, Huang Tiejun, Gao Wen, Tian Qi. Learning Affective Features with a Hybrid Deep Model for Audio-Visual Emotion Recognition. IEEE Transactions on Circuits and Systems for Video Technology, DOI: 10.1109/TCSVT.2017.2719043, 2017.
