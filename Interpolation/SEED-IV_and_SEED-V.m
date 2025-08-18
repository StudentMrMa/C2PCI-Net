


clc,clear;
load('loc62.mat');
load('nan32list.mat');
load('merged_data.mat');
x = loc(:,1);
y = loc(:,2);
PSD_5bands = mergedData;
theta = zeros(37575,32);
alpha = zeros(37575,32);
slow_alpha = zeros(37575,32);
beta = zeros(37575,32);
gamma = zeros(37575,32);
PSD_features = zeros(37575,5,32,32);


    theta = log(PSD_5bands(:,1:62));
    alpha = log(PSD_5bands(:,63:124));
    slow_alpha = log(PSD_5bands(:,125:186));
    beta = log(PSD_5bands(:,187:248));
    gamma = log(PSD_5bands(:,249:310));

for k=1:1:37575
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
for s=1:1:15
a = squeeze(PSD_features((2505*s-2504):2505*s,k,:,:));
size(a)
a(find(~isnan(a))) = (a(find(~isnan(a))) - mean(a(find(~isnan(a)))))/std(a(find(~isnan(a))));
a(find(isnan(a))) = 0;
PSD_features((2505*s-2504):2505*s,k,:,:) = a;
end
end
PSD_features(find(isnan(PSD_features)))=0;


save ('PSD_features', "PSD_features");



ccccc_1 = PSD_featuress(2506:5010,1,:,:);
ccccc_1 = squeeze(ccccc_1)
std(ccccc_1(:))
mean(ccccc_1(:))
save ('ccccc_1.mat' , "ccccc_1");

ccccc_2 = PSD_features(2506:5010,2,:,:);
ccccc_2 = squeeze(ccccc_2)
std(ccccc_2(:))
mean(ccccc_2(:))
save ('ccccc_2.mat' , "ccccc_2");

ccccc_3 = PSD_features(2506:5010,3,:,:);
ccccc_3 = squeeze(ccccc_3)
std(ccccc_3(:))
mean(ccccc_3(:))
save ('ccccc_3.mat' , "ccccc_3");

ccccc_4 = PSD_features(2506:5010,4,:,:);
ccccc_4 = squeeze(ccccc_4)
std(ccccc_4(:))
mean(ccccc_4(:))
save ('ccccc_4.mat' , "ccccc_4");

ccccc_5 = PSD_features(2506:5010,5,:,:);
ccccc_5 = squeeze(ccccc_5)
std(ccccc_5(:))
mean(ccccc_5(:))
save ('ccccc_5.mat' , "ccccc_5");





