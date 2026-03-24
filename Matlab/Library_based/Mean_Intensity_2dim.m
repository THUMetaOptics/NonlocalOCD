function [out] = Mean_Intensity_2dim(I,n)
%UNTITLED2 输入5120*5120的矩阵，按n的批次大小，对二个维度进行平均，并转换成double
%     [a,b] = size(I);
    I = im2double(I);
%     out1 = zeros(size(reshape(I(1,:,:),b,c)));
    for i = 1:5120/n
        for j = 1:5120/n
          I1(i,j) = sum(sum(I(1+(i-1)*n:i*n,1+(j-1)*n:j*n)));
        end
    end
%     b1 = b/10;
    out = I1;
%     out = zeros(b,c);
%     for i = 1:b/10
%         mid = sum(out1((i-1)*10+1:(i-1)*10+10,:));
%         out = out+mid;
%     end

    
end

