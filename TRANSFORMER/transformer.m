clc;
clear 
close all
addpath('..\') %将上一级目录加载进来
addpath(genpath(pwd))
%% 来自：公众号《淘个代码》
X = readmatrix('风电场预测.xlsx');
X = X(3000:end,15);  %只输入功率这一列
n_in = 8;  % 输入前8个时刻的数据
n_out = 1 ; % 此程序为单步预测，因此请将n_out设置为1，否则会报错！
or_dim = size(X,2) ;       % 记录特征数据维度
num_samples = 1000;  % 制作1000个样本。
scroll_window = 1;  %如果等于1，下一个数据从第二行开始取。如果等于2，下一个数据从第三行开始取
[res] = data_collation(X, n_in, n_out, or_dim, scroll_window, num_samples);


% 训练集和测试集划分%% 来自：公众号《淘个代码》

num_size = 0.8;                              % 训练集占数据集比例  %% 来自：公众号《淘个代码》
num_train_s = round(num_size * num_samples); % 训练集样本个数  %% 来自：公众号《淘个代码》

%% 以下几行代码是为了方便归一化，一般不需要更改！
P_train = res(1: num_train_s,1);
P_train = reshape(cell2mat(P_train)',n_in*or_dim,num_train_s);
T_train = res(1: num_train_s,2);
T_train = cell2mat(T_train)';

P_test = res(num_train_s+1: end,1);
P_test = reshape(cell2mat(P_test)',n_in*or_dim,num_samples-num_train_s);
T_test = res(num_train_s+1: end,2);
T_test = cell2mat(T_test)';


%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input);

[t_train, ps_output] = mapminmax(T_train, 0, 1);
t_test = mapminmax('apply', T_test, ps_output);

%%  数据平铺

for i = 1:size(p_train,2)
    trainD{i,:} = (reshape(p_train(:,i),or_dim,[]));
end



for i = 1:size(p_test,2)
    testD{i,:} = (reshape(p_test(:,i),or_dim,[]));
end


targetD =  t_train';
targetD_test  =  t_test';



%% 优化算法优化前，构建优化前的TCN模型


numChannels = or_dim;
maxPosition = 256;
numHeads = 4;
numKeyChannels = numHeads*32;

layers = [ 
    sequenceInputLayer(numChannels,Name="input")
    positionEmbeddingLayer(numChannels,maxPosition,Name="pos-emb");
    additionLayer(2, Name="add")
    selfAttentionLayer(numHeads,numKeyChannels,'AttentionMask','causal')
    selfAttentionLayer(numHeads,numKeyChannels)
    indexing1dLayer("last")
    fullyConnectedLayer(n_out)
    regressionLayer];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,"input","add/in2");




maxEpochs = 50;
miniBatchSize = 32;
learningRate = 0.001;
solver = 'adam';
shuffle = 'every-epoch';
gradientThreshold = 10;
executionEnvironment = "auto"; % chooses local GPU if available, otherwise CPU

options = trainingOptions(solver, ...
    'Plots','none', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', shuffle, ...
    'InitialLearnRate', learningRate, ...
    'GradientThreshold', gradientThreshold, ...
    'ExecutionEnvironment', executionEnvironment);



% 网络训练
tic
net0 = trainNetwork(trainD,targetD,lgraph,options);
toc
analyzeNetwork(net0);% 查看网络结构
%  预测
t_sim1 = predict(net0, trainD); 
t_sim2 = predict(net0, testD); 

%  数据反归一化
T_sim1 = mapminmax('reverse', t_sim1, ps_output);
T_sim2 = mapminmax('reverse', t_sim2, ps_output);
T_train1 = T_train;
T_test2 = T_test;

%  数据格式转换
T_sim1 = double(T_sim1);% cell2mat将cell元胞数组转换为普通数组 %% 来自：公众号《淘个代码》
T_sim2 = double(T_sim2);



% 指标计算 %% 来自：公众号《淘个代码》
disp('…………训练集误差指标…………')
[mae1,rmse1,mape1,error1]=calc_error(T_train1,T_sim1');
fprintf('\n')

figure('Position',[200,300,600,200])
plot(T_train1);
hold on
plot(T_sim1')
legend('真实值','预测值')
title('Transformer训练集预测效果对比')
xlabel('样本点')
ylabel('风速')

disp('…………测试集误差指标…………')
[mae2,rmse2,mape2,error2]=calc_error(T_test2,T_sim2');
fprintf('\n')


figure('Position',[200,300,600,200])
plot(T_test2);
hold on
plot(T_sim2')
legend('真实值','预测值')
title('Transformer预测集预测效果对比')
xlabel('样本点')
ylabel('风速')

figure('Position',[200,300,600,200])
plot(T_sim2'-T_test2)
title('Transformer误差曲线图')
xlabel('样本点')
ylabel('风速')
%% 来自：公众号《淘个代码》