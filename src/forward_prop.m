function a = forward_prop(a,inputs,W,b,activation)
a{1} = inputs;
for i = 2:length(W)
    a{i} = activation(a{i-1},W{i},b{i});
end
end