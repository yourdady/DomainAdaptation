% Transfer Feature Learning with Joint Distribution Adaptation.  
% M. Long, J. Wang, G. Ding, J. Sun, and P.S. Yu.
% IEEE International Conference on Computer Vision (ICCV), 2013.

% Contact: Mingsheng Long (longmingsheng@gmail.com)

clear all;

% Set algorithm parameters
options.k = 100;
options.lambda = 0.1;
options.ker = 'primal';     % 'primal' | 'linear' | 'rbf'
options.gamma = 1.0;        % kernel bandwidth: rbf only
T = 10;

result = [];
for dataStr = {'USPS_vs_MNIST','MNIST_vs_USPS'}
% for dataStr = {'COIL_1','COIL_2'}

    % Preprocess data using L2-norm
    data = strcat(char(dataStr));
    options.data = data;
    load(strcat('../data/',data));
    X_src = X_src*diag(sparse(1./sqrt(sum(X_src.^2))));
    X_tar = X_tar*diag(sparse(1./sqrt(sum(X_tar.^2))));
    
    % 1NN evaluation
    Cls = knnclassify(X_tar',X_src',Y_src,1);
    acc = length(find(Cls==Y_tar))/length(Y_tar); fprintf('NN=%0.4f\n',acc);

    % JDA evaluation
    Cls = [];
    Acc = []; 
    for t = 1:T
        fprintf('==============================Iteration [%d]==============================\n',t);
        [Z,A] = JDA(X_src,X_tar,Y_src,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(X_src,2));
        Zt = Z(:,size(X_src,2)+1:end);

        Cls = knnclassify(Zt',Zs',Y_src,1);
        acc = length(find(Cls==Y_tar))/length(Y_tar); fprintf('JDA+NN=%0.4f\n',acc);
        Acc = [Acc;acc];
    end
    result = [result;Acc(end)];
    fprintf('\n\n\n');
end
fid = fopen(strcat('../result/JDA-Image.o'),'wt');
fprintf(fid,'%0.4f\n',result);
fclose(fid);
