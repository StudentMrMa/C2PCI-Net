clear;
clc;


addpath(genpath(''));
addpath(genpath(''));


filenames_1=get_filenames('E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eeg_raw_data/1/');
filenames_2=get_filenames('E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eeg_raw_data/2/');
filenames_3=get_filenames('E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eeg_raw_data/3/');

eeg_s1_path='E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eeg_raw_data/1/';
eeg_s2_path='E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eeg_raw_data/2/';
eeg_s3_path='E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eeg_raw_data/3/';

eye_s1_path='E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eye_feature_smooth/1/';
eye_s2_path='E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eye_feature_smooth/2/';
eye_s3_path='E:/My_dataset/SEED_IV_dataset/SEED_IV_Original_Data/eye_feature_smooth/3/';


subject_name={'cz','ha','hql','ldy','ly','mhw','mz','qyt','rx','tyc','whh','wll','wq','zjd','zjy'};

subject_number=15;
seg_len=4;
sampling_rate=200;
f_mapsize=32;
ica_flag=1;

de_nan=zeros(1,subject_number);
entropy_stats_nan=zeros(1,subject_number);



for p=1:1
    

    eeg_allband_feature_map=[];
    eeg_psd=[];
    eeg_en_stat=[];
    eye_feature=[];


    for s=1:3 
       
        eval([' load([eeg_s',num2str(s),'_path,filenames_',num2str(s),'{p}]);  ']);
        
        eval([' load([eye_s',num2str(s),'_path,filenames_',num2str(s),'{p}]);  ']);
        
        
        for t=1:24 
            eval([' trial_eeg = ',subject_name{p},'_eeg',num2str(t),';  ']);
            
            trial_eeg=trial_eeg(:,1:end-1);)
            seg_cnt=(size(trial_eeg,2)/(seg_len*sampling_rate));
            seg_cnt=floor(seg_cnt);
            trial_eeg=rereference(trial_eeg');
            trial_eeg=bandpass_filtering(trial_eeg,sampling_rate,4,45);

            if ica_flag==1
                trial_eeg=eeg_max_ica_filtering(trial_eeg);
            end

            eeg_x=eeg_feature_extraction(trial_eeg,sampling_rate,seg_len,ica_flag);

 
            eval([' eye_x=eye_',num2str(t),'; ']);
            eye_x=eye_x';
            x=[eeg_x,eye_x];


            % 特征图构造
            segment_allband_feature_map=zeros(5,f_mapsize,f_mapsize,seg_cnt);
            for i=1:5
                eeg_feature=eeg_x(:,(i-1)*62+1:i*62); 
                eeg_feature_map=spatial_map(eeg_feature,f_mapsize,'./channels62new.loc',1); 

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

        save_name=['E:\My_dataset\SEED_IV_dataset\SEED-IV_PSD/s0',num2str(p),'.mat'];
    else
        save_name=['E:\My_dataset\SEED_IV_dataset\SEED-IV_PSD/s',num2str(p),'.mat'];
    end


    if sum(sum(isnan(eeg_psd)))~=0
        de_nan(1,p)=1;
    end
    

    if sum(sum(isnan(eeg_en_stat)))~=0
        entropy_stats_nan(1,p)=1;
    end


    save(save_name,'eeg_allband_feature_map','eeg_psd','eeg_en_stat','eye_feature');

    clc;

end
