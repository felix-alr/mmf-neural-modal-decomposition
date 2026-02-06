% Training of dlnet-type neural network for mode decomposition

clear all
close all
%% load dataset



useVGG = false;
train = true;
transferLearning = false;
data50k = false;

%  define the input and output size for neural network
N = 5;
input_size = 32;
output_size = 2*N -1;

if useVGG
    model = "VGG";
else
    model = "MLP";
end

%  load the dataset
if N == 3
    dataset = load("data\mmf_3modes_dataset.mat");
elseif data50k
    dataset = load("data\mmf_5modes_dataset_50k.mat");
else
    dataset = load("data\mmf_5modes_dataset.mat");
end

XTrain = dataset.XTrain;
YTrain = dataset.YTrain;
XValid= dataset.XValid;
YValid = dataset.YValid;
XTest = dataset.XTest;
YTest= dataset.YTest;

%% create MLP neural network - step 3 
Layers_MLP = [
            imageInputLayer([input_size input_size 1])
            fullyConnectedLayer(input_size*input_size)
            leakyReluLayer()
            fullyConnectedLayer(input_size*input_size)
            leakyReluLayer()
            fullyConnectedLayer(output_size)];

dlMLP = dlnetwork(Layers_MLP);


%% create VGG neural network - step 7
Layers_VGG= [
    imageInputLayer([input_size input_size 1])

    convolution2dLayer(3,64,"Name","conv1_1","Padding","same")
    reluLayer("Name","relu1_1")
    convolution2dLayer(3,64,"Name","conv1_2","Padding","same")
    reluLayer("Name","relu1_2")
    maxPooling2dLayer(2,"Stride",2,"Name","pooling1")

    convolution2dLayer(3,128,"Name","conv2_1","Padding","same")
    reluLayer("Name","relu2_1")
    convolution2dLayer(3,128,"Name","conv2_2","Padding","same")
    reluLayer("Name","relu2_2")
    maxPooling2dLayer(2,"Stride",2,"Name","pooling2")

    convolution2dLayer(3,256,"Name","conv3_1","Padding","same")
    reluLayer("Name","relu3_1")
    convolution2dLayer(3,256,"Name","conv3_2","Padding","same")
    reluLayer("Name","relu3_2")
    maxPooling2dLayer(2,"Stride",2,"Name","pooling3")
    
    fullyConnectedLayer(256,"Name","fc1")
    fullyConnectedLayer(128,"Name","fc2")
    fullyConnectedLayer(output_size,"Name","fc_output")
];

dlVGG = dlnetwork(Layers_VGG);


if train
    if transferLearning && useVGG & N == 5
        if data50k
            load("data/trained-nets/VGG5modes-transfer-learning.mat");
            disp("Loaded VGG trained on 5 modes using transfer learning.")
        else
            load("data/trained-nets/VGG3modes.mat");
            disp("Loaded VGG trained on 3 modes for transfer learning.")
            graph = removeLayers(layerGraph(dlnet), 'fc_output');
            graph = addLayers(graph, fullyConnectedLayer(output_size, 'Name', 'fc_output'));
            graph = connectLayers(graph, 'fc2', 'fc_output');

           dlnet = dlnetwork(graph);
           disp("Replaced output layer to match the required output size of " + output_size + ".")
        end
    elseif useVGG
        dlnet = dlVGG;
        disp("Initialized untrained VGG for training.")
    else
        dlnet = dlMLP;
        disp("Initialized untrained MLP for training.")
    end
else
   load("data/trained-nets/"+model+"" + N + "modes.mat"); 
   disp("Loaded " + model + " trained on " + N + " mode data.")
end


if train
    %% learnable parameters transfer  - step 8 & 9
    % use Transfer Learning
    

    %% Training network  - step 3
    % define hyperparameters
    % miniBatchSize = 128;
    % numEpochs = 10;
    % learnRate = 0.001;
    options = trainingOptions("adam",Plots="training-progress", ExecutionEnvironment="auto");
    options.MaxEpochs = 10;
    options.InitialLearnRate = 0.001;
    options.MiniBatchSize = 128;


    numObservations = size(XTrain,4);   
    numIterationsPerEpoch = floor(numObservations./options.MiniBatchSize);
    executionEnvironment = "parallel";  

    options.ValidationFrequency = floor(numIterationsPerEpoch./2);
      
    
    %Visualize the training progress in a plot.
    plots = "training-progress";
    % Train Network
    if plots == "training-progress"
        figure
        lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
        lineLossValid = animatedline('Color','#0077b6');
        ylim([0 inf])
        xlabel("Iteration")
        ylabel("Loss")
        yscale("log")
        grid on
        legend([lineLossTrain, lineLossValid], ...
           ["Training Loss", "Validation Loss"])
    end
    iteration = 0;
    start = tic();
    % Train Network
    % Initialize the average gradients and squared average gradients.
    averageGrad = [];
    averageSqGrad = [];
    for epoch = 1:options.MaxEpochs
        
        for i = 1:numIterationsPerEpoch
            iteration = iteration + 1;
            
            % 1. Read mini-batch of data and convert the labels to dummy
            % variables.
            dlX = XTrain(:,:,1,((i-1)*options.MiniBatchSize+1):(i*options.MiniBatchSize));
            labels = YTrain(:, ((i-1)*options.MiniBatchSize+1):(i*options.MiniBatchSize));
            
            % 2. Convert mini-batch of data to a dlarray.
            dlX = dlarray(single(dlX), 'SSCB');
            labels = dlarray(single(labels));
    
            % 3. Evaluate the model gradients and loss using the
            % modelGradients() and dlfeval()
            [grad, loss] = dlfeval(@modelGradients, dlnet, dlX, labels);
    
            % 4. Update the network parameters using the Adam optimizer.
            [dlnet, averageGrad, averageSqGrad] = adamupdate(dlnet, grad, averageGrad, averageSqGrad, iteration, options.InitialLearnRate);
    
    
            % Display the training progress.
            if plots == "training-progress"
                D = duration(0,0,toc(start),'Format','hh:mm:ss');
                addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
    
                if mod(iteration-1, options.ValidationFrequency)==0
                    % Show validate
                    validX = dlarray(single(XValid), 'SSCB');
                    validY = dlarray(single(YValid));
                    [~, loss] = dlfeval(@modelGradients, dlnet, validX, validY);
                    addpoints(lineLossValid,iteration,double(gather(extractdata(loss))))
                end
    
                title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + num2str(double(gather(extractdata(loss)))));
                drawnow
            end
        end
    end
end
%% Test Network  - step 4
% transfer data to dlarray
dlXTest = dlarray(single(XTest), 'SSCB');

% use command "predict"
pred = predict(dlnet, dlXTest);
% use command "extractdata" to extract data from dlarray
pred = extractdata(pred);

% reconstruct field distribution
[Image_data_complex, complex_vector_N] = mmf_rebuilt_image(pred, XTest, N);

%%  Visualization results - step 5
% calculate Correlation between the ground truth and reconstruction
% calculate std
% plot()
% calulate relative error of ampplitude and phase 
corr = zeros(size(complex_vector_N,2),1);

for i = 1:size(corr,1)

    corr(i) = (corr2(squeeze(Image_data_complex(:,:,:,i)), squeeze(XTest(:,:,:,i))));

end

avg_corr = mean(corr);

fprintf("Average correlation on test dataset: %.1f %%\n", avg_corr * 100);

std_dev = std(corr);
fprintf("Standard deviation of the correlation: %.3f %%\n", std_dev * 100);

figure 
boxchart(corr);
grid on
title("Correlation")
ylabel("Correlation")

%%% --- Error analysis (median) -------------------------------------------

% Calculation of relative error of amplitude and phase 

% Predicted rho and phase extracted from predicted_YTest
pred_ampl = pred(1:N,:);
pred_phi = pred(N + 1:N* 2 - 1,:);

% Real rho and phase extracted from Y_Test
ampl = YTest(1:N,:);
phi  = YTest(N+1:N*2-1,:);

% For excluding zeros (avoid division by zero)
minimumNumber = 1e-12; 

rel_err_ampl        = abs(pred_ampl - ampl) ./ max(abs(ampl), minimumNumber);
median_rel_err_ampl = median(rel_err_ampl, 'all');

rel_err_phi      = abs(pred_phi - phi) ./ max(abs(phi),minimumNumber);
median_rel_err_phi = median(rel_err_phi,'all');

fprintf("Median relative error of amplitude: %.1f %%\n",median_rel_err_ampl * 100);
fprintf("Median relative error of phase: %.1f %% \n",median_rel_err_phi * 100);


%% save model
if transferLearning
    transfer = "-transfer-learning";
else
    transfer = "";
end

if data50k
    transfer = transfer+"50k";
end

save("data/trained-nets/"+model+"" + N + "modes" + transfer + ".mat", "dlnet");