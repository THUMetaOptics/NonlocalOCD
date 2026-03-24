function I = Select_circle(I,radius,center_x,center_y)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
x0 = center_x; % 圆心的x坐标
y0 = center_y; % 圆心的y坐标
r = radius;   % 圆的半径
% 创建网格
[X, Y] = meshgrid(1:256, 1:256);
% 计算每个像素到圆心的距离
distance = sqrt((X - x0).^2 + (Y - y0).^2);
% 创建掩码
mask = distance <= r;
% 提取圆形区域的像素
I = I .* mask;
end

