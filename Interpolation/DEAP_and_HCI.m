


clc,clear;

x = loc(:,1);
y = loc(:,2);
PSD_5bands = combined19200x32_data;
theta = zeros(19200,32);
alpha = zeros(19200,32);
slow_alpha = zeros(19200,32);
beta = zeros(19200,32);
gamma = zeros(19200,32);
PSD_features = zeros(19200,5,32,32);

% 取出各个频段的数据
    theta = log(PSD_5bands(:,1:32));
    alpha = log(PSD_5bands(:,33:64));
    slow_alpha = log(PSD_5bands(:,65:96));
    beta = log(PSD_5bands(:,97:128));
    gamma = log(PSD_5bands(:,129:160));

for k=1:1:19200
[X1,Y1,Z1] = griddata(x,y,theta(k,:),linspace(-0.5179,0.5179,32)',linspace(-0.5179,0.5179,32),'v4');
[X2,Y2,Z2] = griddata(x,y, alpha(k,:),linspace(-0.5179,0.5179,32)',linspace(-0.5179,0.5179,32),'v4');
[X3,Y3,Z3] = griddata(x,y, slow_alpha(k,:),linspace(-0.5179,0.5179,32)',linspace(-0.5179,0.5179,32),'v4');
[X4,Y4,Z4] = griddata(x,y, beta(k,:),linspace(-0.5179,0.5179,32)',linspace(-0.5179,0.5179,32),'v4');
[X5,Y5,Z5] = griddata(x,y,gamma(k,:),linspace(-0.5179,0.5179,32)',linspace(-0.5179,0.5179,32),'v4');

Z1 = flipud(Z1);
Z2 = flipud(Z2);
Z3 = flipud(Z3);
Z4 = flipud(Z4);
Z5 = flipud(Z5);


Z1(nan32_list)=NaN;
Z2(nan32_list)=NaN;
Z3(nan32_list)=NaN;
Z4(nan32_list)=NaN;
Z5(nan32_list)=NaN;

PSD_features(k,1,:,:) = Z1;
PSD_features(k,2,:,:) = Z2;
PSD_features(k,3,:,:) = Z3;
PSD_features(k,4,:,:) = Z4;
PSD_features(k,5,:,:) = Z5;

disp(k)
end
for k=1:1:5
for s=1:1:32
a = squeeze(PSD_features((600*s-599):600*s,k,:,:));
size(a)
a(find(~isnan(a))) = (a(find(~isnan(a))) - mean(a(find(~isnan(a)))))/std(a(find(~isnan(a))));
a(find(isnan(a))) = 0;
PSD_features((600*s-599):600*s,k,:,:) = a;
end
end
PSD_features(find(isnan(PSD_features)))=0;

save ('PSD_features', "PSD_features");
ccccc = PSD_features(601:1200,3,:,:);
ccccc = squeeze(ccccc)
std(ccccc(:))
mean(ccccc(:))
save ('ccccc.mat' , "ccccc");


