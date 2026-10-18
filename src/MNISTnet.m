[train_inputs,train_labels] = load_MNIST("train-images.idx3-ubyte","train-labels.idx1-ubyte");
[test_inputs,test_labels] = load_MNIST("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte");
num_samples = size(train_labels,2);
train_targets = zeros(10,num_samples);
train_targets(sub2ind(size(train_targets),train_labels+1,1:num_samples)) = 1; % one-hot encoding
pixels = numel(train_inputs(:,:,1));
train_inputs = train_inputs./255; %input normalisation
train_data = [reshape(train_inputs,[pixels,num_samples]);train_targets]; 
% split into training and validation set
train_set = train_data(:,10001:end); 
valid_set = train_data(:,1:10000);

% test set
num_samples = size(test_labels,2);
test_targets = zeros(10,num_samples);
test_targets(sub2ind(size(test_targets),test_labels+1,1:num_samples)) = 1; % one-hot encoding
test_inputs = test_inputs./255; %input normalisation
test_data = [reshape(test_inputs,[pixels,num_samples]);test_targets];

architecture = [pixels,50,10]; % network architecture
maxEpoch = 100; % maximum number of epochs
batch_size = 200; % batch size
activation = @leaky_relu; % activation function
learning_rate = @learning_rate; % learning rate schedule
loss = 'softmax'; % loss function
lambda = 0.05;

[W,b,cost,acc] = neural_net(architecture,train_set,valid_set,activation,...
    loss,learning_rate,maxEpoch,batch_size,lambda);

%confusion matrix for test data
a = cell(1,length(architecture));
a = forward_prop(a,test_data(1:end-10,:),W,b,activation);
a = exp(a{end})./sum(exp(a{end}),1);
[~,classification] = max(a,[],1);
classification = classification-1;
prediction = categorical(classification);
true = categorical(test_labels);
%figure
plotconfusion(true,prediction)