%%--------------------------------------------------------------------------
% Reference:
%
% Quanxue Gao, Pu Zhang, Wei Xia, De{-}Yan Xie, Xinbo Gao, Dacheng Tao:
% Enhanced Tensor RPCA and Its Application.
% IEEE TPAMI 2020, doi: 10.1109/TPAMI.2020.3017672
%
% version 1.0 --Jun./2020
%
%--------------------------------------------------------------------------
% Written by (xd.weixia@gmail.com)
%--------------------------------------------------------------------------

%% Load Setting
clc;
clear all;
addpath([pwd, '/Dataset']);
addpath([pwd, '/funs']);

%% Load Dataset
load('ori');                                % raw sample
load('ori_noise_10');                       % noisy sample
Nun_S = length(ori);
[n1,n2] = size(ori{1,1});                   % sample size
n=min(n1,n2);

result_psnr = [];
result_ssim = [];

%% Hyper-Parameters
% Weighted vector of weighted tensor Schatten p-norm
w = [];
w = [w; 1*ones(10,1)];
w = [w; 1.1*ones(70,1)];
w = [w; 1.5*ones(n-80,1)];

% The power p of weighted tensor Schatten p-norm£¬p in (0,1]
p=0.9;

%% Optimizataion
% One sample at a time
for i = 10 % 1:1:Nun_S
    disp('---------------------------------------------------------');
    disp(['Processing the ' num2str(i) '-th sample:']);
    ori_temp = ori{i};
    X = ori_noise_10{i};
    [D, E, obj, err, iter]  = etrpca_tnn_lp(X, 1/sqrt(481*3), w, p);
    
    % Cal. PSNR
    temp = [];
    temp = [temp PSNR( ori_temp, D)];
    result_psnr = [result_psnr; temp];
    
    % Cal. SSIM
    temp1 = [];
    temp1 = [temp1 ssim_index(ori_temp, D)];
    result_ssim = [result_ssim; temp1];
    
    % Record Results
    fid = fopen('.\ETRPCA.txt','a+');
    fprintf(fid,'The results of the %g-th sample:', i);
    fprintf(fid,'%15s', ['PSNR£º', num2str(temp)], ['SSIM£º', num2str(temp1)]);
    fprintf(fid,'\n');
    
    % Visualization
    figure(i);
    subplot(1,2,1);
    imshow(ori_temp/255,[]);              % Raw sample
    subplot(1,2,2);
    imshow(D/255,[]);                     % Recovered sample
end