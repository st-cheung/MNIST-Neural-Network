function delta = backprop(delta,a,labels,W,activation,loss)
L = length(delta);
switch loss
    case 'softmax'
        shift = a{L}-max(a{L});
        deltaC = exp(shift)./sum(exp(shift),1)-labels;
    case 'lstsq'
        deltaC = a{L}-labels;
    otherwise
        warning('Unexpected loss function.')
end
delta{end} = activation(a{L}).*deltaC;
for i = L-1:-1:2
    delta{i} = activation(a{i}).*(W{i+1}'*delta{i+1});
end
end