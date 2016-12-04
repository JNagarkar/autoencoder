%% Product of Elements in Each Plane  

%% 
% Create a 3-by-3-by-2 array whose elements correspond to their linear indices. 
A=[1:3:7;2:3:8;3:3:9];
A(:,:,2)=[10:3:16;11:3:17;12:3:18]  

%% 
% Find the product of each element in the first plane with its
% corresponding element in the second plane. The length of the first
% dimension matches |size(A,1)|, the length of the second dimension matches
% |size(A,2)|, and the length of the third dimension is 1.
dim = 3;
B = prod(A,dim) 
