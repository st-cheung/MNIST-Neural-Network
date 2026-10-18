function [images, labels] = load_MNIST(image_file,label_file)
file = fopen(image_file,'r','b');
meta_data = num2cell(fread(file,4,'int32'));
[check,num_items,rows,cols] = meta_data{:};
if check ~= 2051
    error("Incompatible image file format")
end
images = zeros(rows, cols, num_items);
for n = 1:num_items
    for i = 1:rows
        images(i,:,n) = fread(file,cols,'uint8');
    end
end
fclose(file);
file = fopen(label_file, 'r','b');
meta_data = fread(file,2,'int32');
if meta_data(1) ~= 2049
    error("Incompatible label file format")
end
labels = fread(file,'uint8')';
fclose(file);
end

