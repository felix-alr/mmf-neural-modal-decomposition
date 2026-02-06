function [image_data] = mmf_build_image(number_of_modes,image_size,number_of_data, complex_weights_vector)
%% load complex mode distribution
% load the complex distrbutions 
% mmf_3modes_32 for 3-mode case 
% mmf_5modes_32 for 5-mode case
if number_of_modes == 3   
    load('mmf_3modes_32.mat')
    mmf_modes = mmf_3modes_32;
elseif number_of_modes == 5
    load('mmf_5modes_32.mat')
    mmf_modes = mmf_5modes_32;
end
%% create images
% define a variable for Image data with dimension (image size, image size, 1, n)
fprintf("Start to generate the mode distribution......\n");

image_data = zeros(image_size,image_size,1,number_of_data);

for index=1:number_of_data
    % 1. define a variable for single image with resolution (image size,image size)
    img = zeros(image_size, image_size);
    % 2. generation of complex field distribution 
    for m=1:number_of_modes
        img = img + mmf_modes(:,:,m).* complex_weights_vector(index, m);
    end
    % 3. abstract Amplitude distribution 
    img = abs(img);
    % 4. normalization the amplitude distribution to (0,1)
    img = normalization(img, 0, 1);
    image_data(:,:,1,index) = img;
    
end
fprintf("The image data has been generated.\n");

end

