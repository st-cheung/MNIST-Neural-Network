function activation = leaky_relu(a,varargin) 
if nargin>1
    activation = varargin{1}*a+varargin{2};
    activation = (activation>0).*activation+0.01.*(activation<=0).*activation;
else
   activation = (a>0)+0.01.*(a<=0);
end
end