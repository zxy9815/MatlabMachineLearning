function [faces] = load_faces()    
    % Data Parameters
    num_people = 40;       % # People in dataset
    num_img_pp = 10;       % # images per person in each subdirectory
    img_fmt = '.pgm';      % image format (portable grayscale map)
    img_size = [112,92];   % image size (rows,columns)
  
    % Load data from directory into workspace
    faces = zeros(num_people*num_img_pp,prod(img_size));
    for person = 1:num_people
        for img = 1:num_img_pp
            img_path = strcat('./att-database-of-faces/s',num2str(person), ...
                        '/', num2str(img), img_fmt);
            person_img = imread(img_path);
            faces((person-1)*num_img_pp+img,:) = person_img(:)'; 
        end
    end
end