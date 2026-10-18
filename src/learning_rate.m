function eta = learning_rate(epochs)
if epochs <= 90
    eta = 0.5;
else
    eta = 0.05;
end
