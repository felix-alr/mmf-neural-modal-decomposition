%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 5;    %option: 3 or 5
number_of_data = 50000;
image_size = 32;    % resolution 32x32
 
%% generation of complex mode weights and label vector - step 1

% 1. create random amplitude weights. The weights of amplitude should be normalized.
rho = rand(number_of_data,number_of_modes);
rho_n= rho./vecnorm(rho, 2, 2);

% 2. create random phase amplitude. (Using realtive phase difference)
phi = rand(number_of_data,number_of_modes)*2*pi-pi;
phi_rel = (phi(:, 2:end)-phi(:, 1));

% 3. complex mode weights vector
k_i=rho_n.*exp(1i*phi);

% 4. normalize cos(phase) to (0,1)
phi_rel_n = (cos(phi_rel)+1)/2;

% 5. combine amplitude and phase into a label vector (1,2N-1)
label = [rho_n, phi_rel_n];

% 6. split complex mode weights vector and label vector into Training,
% validation and test set. 
num_train = floor(0.7 * number_of_data);
num_val = floor(0.15 * number_of_data);

% Split labels
YTrain = transpose(label(1:num_train, :));
YValid = transpose(label(num_train+1:num_train+num_val, :));
YTest = transpose(label(num_train+num_val+1:end, :));

% Split weights
k_train = k_i(1:num_train, :);
k_val = k_i(num_train+1:num_train+num_val, :);
k_test = k_i(num_train+num_val+1:end, :);

%% create image data - step 2
% use function mmf_build_image()
XTrain = mmf_build_image(number_of_modes,image_size,size(k_train, 1),k_train);
XValid = mmf_build_image(number_of_modes,image_size,size(k_val, 1),k_val);
XTest = mmf_build_image(number_of_modes,image_size,size(k_test, 1),k_test);

%% save dataset

save(strcat(strcat("data/mmf_", num2str(number_of_modes)), "modes_dataset_50k.mat"), "XTrain", "YTrain", "XValid", "YValid", "XTest", "YTest");

fprintf('Dataset saved successfully.\n')