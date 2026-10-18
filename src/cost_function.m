function [acc,cost] = cost_function(data,label,W,b,a,activation,loss,lambda)
a = forward_prop(a,data,W,b,activation);
[max_output,predicted_label] = max(a{end},[],1);
[~,true] = max(label,[],1);
acc = sum(predicted_label==true)/length(data);
switch loss
    case 'softmax'
        shift = a{end}-max_output;
        index = sub2ind(size(a{end}),true,1:length(data));
        soft_max = exp(shift(index))./sum(exp(shift),1);
        penalty = 0;
        if lambda 
            for l = 1:length(W)
                penalty = penalty + sum(W{l}.^2,'all');
            end
        end
        cost = (-sum(log(soft_max+realmin))+ lambda*penalty)/length(data);
    case 'lstsq'
        cost = sum(sum((label-a{end}).^2))/length(data);
end
end