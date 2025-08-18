

% author ：Mazhuang
% time : 2023.12.08




clc;
clear;
close all;

addpath(genpath('../matlab_preprocess/'));
addpath(genpath('./eeglab2022.0/'));

sampling_rate = 128;
seg_len = 4; 
low_cutoff = 4;
high_cutoff = 45;
ica_flag = 1;

subject_count = 30;
trial_count = 20;
saved_subject_count = 0;

for i = 1:subject_count
    all_eeg_psd = [];
    all_eeg_en_stat = [];
    all_peri_feature = [];

    for j = 1:trial_count
        a1 = int2str(i);
        b1 = int2str(j);
        raw_path = '';
        name = strcat(raw_path,'Part_',a1,'_S_Trial',b1,'_emotion.mat');
        if ~isfile(name)
            continue;
        end
       
        data = matfile(name);
        data = data.data; 
        data = squeeze(data);
        data = data';
        data = downsample(data,2);
        data = data';
        data = double(data);
        data = data(:,sampling_rate*30+1:end-sampling_rate*5);


        trial_eeg = data(1:32,:); 
        trial_eeg = rereference(trial_eeg'); 
        trial_eeg = bandpass_filtering(trial_eeg, sampling_rate, low_cutoff, high_cutoff);
        if ica_flag == 1
            trial_eeg = eeg_max_ica_filtering(trial_eeg);
        end


        power_4_8 = extract_psd(trial_eeg, sampling_rate, seg_len, 4, 8); 
        power_8_10 = extract_psd(trial_eeg, sampling_rate, seg_len, 8, 10); 
        power_8_12 = extract_psd(trial_eeg, sampling_rate, seg_len, 8, 12); 
        power_12_30 = extract_psd(trial_eeg, sampling_rate, seg_len, 12, 30); 
        power_30_45 = extract_psd(trial_eeg, sampling_rate, seg_len, 30, 45);
        [entropy_shannon, entropy_log_energy] = extract_entropy(trial_eeg, sampling_rate, seg_len);
        [mean_eeg, variance, zero_crossing_rate, kurtosis_eeg, skewness_eeg] = extract_statistic(trial_eeg, sampling_rate, seg_len);


        eeg_psd = [power_4_8 power_8_10 power_8_12 power_12_30 power_30_45];
        eeg_en_stat = [entropy_shannon entropy_log_energy mean_eeg variance zero_crossing_rate kurtosis_eeg skewness_eeg];


        peri_data = data(33:end,:); 
        trial_peri = peri_data';
        peri_x = peri_feature_extraction_HCI_2(trial_peri, sampling_rate, seg_len);



        all_eeg_psd = [all_eeg_psd; eeg_psd];
        all_eeg_en_stat = [all_eeg_en_stat; eeg_en_stat];
        all_peri_feature = [all_peri_feature; peri_x];
    end

    if ~isempty(all_eeg_psd)
        saved_subject_count = saved_subject_count + 1;

        eeg_psd = all_eeg_psd;
        eeg_en_stat = all_eeg_en_stat;
        peri_feature = all_peri_feature;

        save_name = sprintf('s%d.mat', saved_subject_count);
        save_path = '';
        save(fullfile(save_path, save_name), 'eeg_psd', 'eeg_en_stat', 'peri_feature');
    end
end



