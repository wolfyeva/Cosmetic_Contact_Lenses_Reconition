imds = imageDatastore('D:\WolfyEva\Matlab\Project','IncludeSubfolders',true,'LabelSource','foldernames')
figure;
perm = randperm(1000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end
labelCount = countEachLabel(imds)
img = readimage(imds,1);
size(img)
RGB=imread('With_CCL\WithCCL_11_IrisRight.bmp');
win1 = centerCropWindow2d(size(RGB),[255 255]);
B1 = imcrop(RGB,win1);
imshow(B1);


%numTrainFiles = 1000;
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.3,'randomize');

layers = [
    imageInputLayer([255 255 1])
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    
    convolution2dLayer(3,256,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
imageSize = layers(1).InputSize;

augTrain = augmentedImageDatastore(imageSize,imdsTrain,'ColorPreprocessing','rgb2gray','OutputSizeMode','centercrop');
augValidation = augmentedImageDatastore(imageSize,imdsValidation,'ColorPreprocessing','rgb2gray','OutputSizeMode','centercrop');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
analyzeNetwork(layers);
net = trainNetwork(augTrain,layers,options);
net.Layers(end)
trainingFeatures = activations(net, augTrain, 'classoutput', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
classifier = fitcecoc(trainingFeatures, imdsTrain.Labels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testImage = readimage(imdsValidation,1);
testLabel = imdsValidation.Labels(1)
ds = augmentedImageDatastore(imageSize, testImage,'ColorPreprocessing','rgb2gray','OutputSizeMode','centercrop');
imageFeatures = activations(net, ds, 'classoutput', 'OutputAs', 'columns');
predictedLabel = predict(classifier, imageFeatures, 'ObservationsIn', 'columns')

YPred = classify(net,augValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)