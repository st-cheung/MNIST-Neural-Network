function [W,b,cost_vec,acc_vec] = neural_net(architecture,train_set,valid_set,activation,loss,learning_rate,num_epoch,batch_size,lambda)
    layers = length(architecture);
    if (architecture(1)+architecture(end)~=size(train_set,1))
        error('Architecture does not match data.')
    end
    [W,b,a,delta] = deal(cell(1,layers)); 
    acc_vec = zeros(2,num_epoch+1);
    cost_vec = zeros(1,num_epoch+1);
    dW = W; db = b;
    % initialisation of weights and biases
    for l = 2:layers
        W{l} = randn(architecture(l),architecture(l-1))*0.5;
        b{l} = randn(architecture(l),1)*0.5;
        dW{l} = zeros(architecture(l),architecture(l-1));
        db{l} = zeros(architecture(l),1);
    end
    input_size = architecture(1);
    num_samples = size(train_set,2);
    num_batches = ceil(num_samples/batch_size);
    [train_acc,cost] = cost_function(train_set(1:input_size,:),train_set(input_size+1:end,:),W,b,a,activation,loss,lambda);
    cost_vec(1) = cost; acc_vec(1,1) = train_acc;
    [valid_acc,~] = cost_function(valid_set(1:input_size,:),valid_set(input_size+1:end,:),W,b,a,activation,loss,lambda);
    acc_vec(2,1) = valid_acc;
    fprintf('Epoch 0: cost = %.4f, training accuracy = %.4f, validation accuracy = %.4f\n', ...
        cost,train_acc,valid_acc);
    for epoch = 1:num_epoch
        shuffle = train_set(:,randperm(num_samples));    
        eta = learning_rate(epoch);
        for iter = 1:num_batches
            batch = shuffle(:,(1+(iter-1)*batch_size):min(iter*batch_size,num_samples));
            % forward pass
            a = forward_prop(a,batch(1:input_size,:),W,b,activation);
            % backward pass
            delta = backprop(delta,a,batch(input_size+1:end,:),W,activation,loss);
            % update weights and biases
            current_batch_size = size(batch,2);
            for l = 2:layers
                gradW = (delta{l}*a{l-1}'+2*lambda*W{l})./current_batch_size;
                gradb = sum(delta{l},2)./current_batch_size;
                dW{l} = eta.*gradW;
                db{l} = eta.*gradb;
                W{l} = W{l} - dW{l};
                b{l} = b{l} - db{l};
            end
        end
        % evaluation of network accuracy and cost function
        [train_acc,cost] = cost_function(train_set(1:input_size,:),train_set(input_size+1:end,:),W,b,a,activation,loss,lambda);
        cost_vec(epoch+1) = cost; acc_vec(1,epoch+1) = train_acc;
        [valid_acc,~] = cost_function(valid_set(1:input_size,:),valid_set(input_size+1:end,:),W,b,a,activation,loss,lambda);
        acc_vec(2,epoch+1) = valid_acc;
        % display training progress
        if mod(epoch, 10) == 0
            fprintf('Epoch %d: cost = %.4f, training accuracy = %.4f, validation accuracy = %.4f\n', ...
                epoch,cost,train_acc,valid_acc);
        end
    end
    % plot error rate/accuracy and cost function
    figure; hold on
    plot(1:epoch,1-acc_vec(1,2:epoch+1),'b-x','LineWidth',0.5)
    plot(1:epoch,1-acc_vec(2,2:epoch+1),'r-o','LineWidth',0.5)
    ylim([0,0.15]);
    xlabel('Epoch Number'); ylabel('Error rate')
    legend("Training Data","Validation Data")
    hold off
    figure;
    semilogy(0:epoch,cost_vec(1,1:epoch+1),'b-x','LineWidth',0.5)
    ylim([0,1]);
    xlabel('Epoch Number'); ylabel('Cost')
    legend("Training Data",'location','northeast')
    hold off
end
