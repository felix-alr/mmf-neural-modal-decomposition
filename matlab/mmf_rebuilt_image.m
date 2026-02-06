function [Image_data_complex,complex_vector_N] = mmf_rebuilt_image(pred,ground_truth,number_of_modes)
% Input>
% pred_vectorts: the prediction from neural network
% ground_truth: the correct amplitude distributions
% Outpu>
% Image_data_complex: reconstructed complex distribution 
% complex_vector_N: complex mode weights for all test data 
if number_of_modes == 3   
    load('phase_variant_3.mat')
    load('mmf_3modes_32.mat')
    mmf_modes = mmf_3modes_32;
elseif number_of_modes == 5
    load('phase_variant_5.mat')
    load('mmf_5modes_32.mat')
    mmf_modes = mmf_5modes_32;
end

pred = transpose(pred);

image_size=size(mmf_modes,1);
number_of_test_images=size(pred,1);
ground_truth = squeeze(ground_truth);
%% read mode weights from predicted vectors
% read amplitude weigths

ampl_weights = pred(:,1:number_of_modes);

% read cos(phase) 
phi_cos_n = pred(:, number_of_modes+1:end);
% normalization cos(phase) to (-1,1)
phi_cos = phi_cos_n .* 2 -1;

% calculate phase through arccos()
% and add phase weight of the first mode(phase value = 0) 
phi = zeros(size(phi_cos,1), number_of_modes);
phi(:,2:number_of_modes) = acos(phi_cos);


%% rebuilt phase vector
% define a varibale for complex vectors
complex_vector_N  = zeros(number_of_test_images,number_of_modes);

for i1=1:number_of_test_images
    % read phase weights and generate all possible combinations
    phi_vectors = phi(i1,:).* phase_weight;
    complex_vector_n = zeros(size(phi_vectors,1),number_of_modes);
    % read the ground truth
    ground_truth_i = ground_truth(:,:,i1);
    correlation_n = zeros(size(phi_vectors,1),1);
    
    % reconstruct all possible field distribution
    for i2 = 1:size(phi_vectors,1)
        complex_vector = ampl_weights(i1,:) .*exp(1i* phi_vectors(i2,:));
        
        % 1. define a variable for single image with resolution (image size,image size)
        img = zeros(image_size, image_size);

        % 2. generation of complex field distribution
        for mode=1:number_of_modes
            img = img + mmf_modes(:,:,mode) * complex_vector(mode);
        end 

        
        % 3. abstract Amplitude distribution
        %    abs(template)
        template = img;
        
        % calculate the correlation coefficient between reconstrion and ground
        % truth
        correlation = abs(corr2(abs(template),ground_truth_i));
        correlation_n(i2) = correlation;
        complex_vector_n(i2,:) = complex_vector;
        
    end       
    % find the right phase weights regarding on the max correlation
    posx = find(correlation_n == max(correlation_n));   
    if numel(posx) > 1
        posx = posx(1);
    end
    complex_vector_N(i1,:) = complex_vector_n(posx,:);
end
%% rebuilt the distribution(complex) 
% using function: mmf_build_image()

Image_data_complex = mmf_build_image(number_of_modes,image_size,number_of_test_images, complex_vector_N);
complex_vector_N = transpose(complex_vector_N);

end
