function [PSNR] = PSNR(X, Y)

 if size(X,3)~=1   
   org=rgb2ycbcr(X);
   test=rgb2ycbcr(Y);
   Y1=org(:,:,1);
   Y2=test(:,:,1);
   Y1=double(Y1);  
   Y2=double(Y2);
 else              
     Y1=double(X);
     Y2=double(Y);
 end
 
if nargin<2    
   D = Y1;
else
  if any(size(Y1)~=size(Y2))
    error('The input size is not equal to each other!');
  end
 D = Y1 - Y2; 
end
MSE = sum(D(:).*D(:)) / numel(Y1); 
PSNR = 10*log10(255^2 / MSE);
