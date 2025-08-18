clear;
clc;

addpath(genpath('../matlab_preprocess'));


addpath './eye_feature/'

addpath ''

filenames=get_filenames('');

subject_number=16;
EEG_channel_idx=[1:32 34:42 44:64];
s1_start_second=[30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204];
s1_end_second=[102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359];

s2_start_second=[30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741];
s2_end_second=[267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817];

s3_start_second=[30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888];
s3_end_second=[321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066];




seg_len=4;
sampling_rate=200;
f_mapsize=32;
ica_flag=1;

de_nan=zeros(1,subject_number);
entropy_stats_nan=zeros(1,subject_number);
eye_nan=zeros(1,subject_number);

for p=1:16


    eeg_allband_feature_map=[];
    eeg_psd=[];
    eeg_en_stat=[];
    eye_feature=[];




    eye_filename=['s',num2str(p),'.mat'];
    load(eye_filename);

    for s=1:3
        filename=filenames{(p-1)*3+s};
        EEG=pop_loadcnt(filename);
        
        session_data=EEG.data(EEG_channel_idx,:); 
        session_data=session_data';
        session_data=rereference(session_data);
        session_data=double(session_data);
        session_data=bandpass_filtering(session_data,sampling_rate,4,45);
        
        if ica_flag==1
            session_data=eeg_max_ica_filtering(session_data,2000);
        end


        for t=1:15

            if s==1
                trial_data=session_data(s1_start_second(t)*sampling_rate:s1_end_second(t)*sampling_rate-1,:);
            elseif s==2
                trial_data=session_data(s2_start_second(t)*sampling_rate:s2_end_second(t)*sampling_rate-1,:);
            else
                trial_data=session_data(s3_start_second(t)*sampling_rate:s3_end_second(t)*sampling_rate-1,:);
            end
            seg_cnt=size(trial_data,1)/(seg_len*sampling_rate);
            seg_cnt=floor(seg_cnt);
            
      
            eeg_x=eeg_feature_extraction(trial_data,sampling_rate,seg_len);
            

        
            eval([' eye_x= eye_s',num2str(s),'_t',num2str(t),';']);
            x=[eeg_x,eye_x];

            
       
            segment_allband_feature_map=zeros(5,f_mapsize,f_mapsize,seg_cnt);
            for i=1:5
                eeg_feature=eeg_x(:,(i-1)*62+1:i*62);
                eeg_feature_map=spatial_map(eeg_feature,f_mapsize,'./channels62new.loc',0);
                segment_allband_feature_map(i,:,:,:)=eeg_feature_map;

            end

            segment_allband_feature_map=map_shape_changing(segment_allband_feature_map,f_mapsize,seg_cnt);
           

            eeg_allband_feature_map=[eeg_allband_feature_map;segment_allband_feature_map];
            eeg_psd=[eeg_psd;eeg_x(:,1:62*5)];
            eeg_en_stat=[eeg_en_stat;eeg_x(:,62*5+1:end)];
            eye_feature=[eye_feature;eye_x];
            


        end
    end


    if p<10
        save_name=['',num2str(p),'.mat'];
    else
        save_name=['',num2str(p),'.mat'];
    end
    
    if sum(sum(isnan(eeg_psd)))~=0
        de_nan(1,p)=1;
    end
    

    if sum(sum(isnan(eeg_en_stat)))~=0
        entropy_stats_nan(1,p)=1;
    end
    

    if sum(sum(isnan(eye_feature)))~=0
        eye_nan(1,p)=1;
    end
    
    save(save_name,'eeg_allband_feature_map','eeg_psd','eeg_en_stat','eye_feature');
    clc;
    
    
end
