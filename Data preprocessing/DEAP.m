clear;
clc;

addpath(genpath(''));
addpath(genpath(''));
addpath(genpath(''));

fs=128;
segment_length=4;
f_mapsize=32;
segment_cnt=60/segment_length;
ica_flag=1;
feature_smooth_lambda=0.7;

startP=1;
stopP=32;

psd_nan=zeros(1,stopP-startP+1);
entropy_stats_nan=zeros(1,stopP-startP+1);
peri_nan=zeros(1,stopP-startP+1);


for p=startP:stopP
    filename=['F:\Data_Preprocessing\DataSet\DEAP_Original_Data\s',num2str(p),'.mat'];
    load(filename);

    eeg_allband_feature_map=zeros(40*segment_cnt,5,f_mapsize,f_mapsize);
    eeg_psd=zeros(40*segment_cnt,32*5);

    eeg_en_stat=zeros(40*segment_cnt,384-32*5);
    peri_feature=zeros(40*segment_cnt,55);


    for t=1:40
        
        trial=data(t,:,:);
        trial=squeeze(trial);
        trial=trial(:,3*fs+1:end);
        trial_eeg=trial(1:32,:);


        trial_peri=trial(33:end,:);


        eeg_x=eeg_psd_feature_extraction(trial_eeg,fs,segment_length,ica_flag);
        peri_x=peri_feature_extraction_DEAP(trial_peri',fs,segment_length);
        x=[eeg_x,peri_x]; 
        

        
        segment_allband_feature_map=zeros(5,f_mapsize,f_mapsize,segment_cnt);
        for i=1:5
            eeg_feature=eeg_x(:,(i-1)*32+1:i*32);

         
            eeg_feature_map=spatial_map(eeg_feature,f_mapsize,'F:\Data_Preprocessing\Channel_Location\Standard-10-20-Cap32.locs',1);

            segment_allband_feature_map(i,:,:,:)=eeg_feature_map;
        end

        segment_allband_feature_map=map_shape_changing(segment_allband_feature_map,f_mapsize,segment_cnt);
        
        eeg_allband_feature_map((t-1)*segment_cnt+1:t*segment_cnt,:,:,:)=segment_allband_feature_map;
        eeg_psd((t-1)*segment_cnt+1:t*segment_cnt,:)=eeg_x(:,1:32*5);

        eeg_en_stat((t-1)*segment_cnt+1:t*segment_cnt,:)=eeg_x(:,32*5+1:end);
        peri_feature((t-1)*segment_cnt+1:t*segment_cnt,:)=peri_x;

    end
    

    if p<10
        save_name=['',num2str(p),'.mat'];
    else
        save_name=['',num2str(p),'.mat'];
    end

    if sum(sum(isnan(eeg_psd)))~=0

        psd_nan(1,p-(stopP-startP+1))=1;
    end
    
    if sum(sum(isnan(eeg_en_stat)))~=0
        entropy_stats_nan(1,p)=1;
    end
    
    if sum(sum(isnan(peri_feature)))~=0
        peri_nan(1,p)=1;
    end
    
    save(save_name,'eeg_allband_feature_map','eeg_psd','eeg_en_stat','peri_feature');
    clc;

end







