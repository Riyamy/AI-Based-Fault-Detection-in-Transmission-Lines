% fault_simulation.m
% Generates synthetic transmission line fault signals and saves to data/raw/

clear; close all; clc;

fs = 2000;             % Sampling frequency
duration = 0.5;        % 0.5 seconds
t = 0:1/fs:duration-1/fs;

out_dir = '../data/raw/';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

labels = {'normal','SLG','LLG','LLL'};   % Fault types
n_per_class = 20; % adjust for dataset size

manifest = {};
idx = 1;

for li = 1:length(labels)
    lab = labels{li};
    for s = 1:n_per_class
        % Base waveform
        V = sin(2*pi*50*t) + 0.01*randn(size(t));
        
        switch lab
            case 'normal'
                sig = V;
            case 'SLG' % Single Line to Ground
                sig = V;
                sig(200:400) = 0.2*sig(200:400);
            case 'LLG' % Line to Line to Ground
                sig = V;
                sig(300:450) = -0.5 + 0.2*randn(1,151);
            case 'LLL' % Three-phase fault
                sig = V + 0.5*sin(2*pi*150*t);
                sig(150:300) = sig(150:300)*0.1;
        end

        fname = sprintf('%s%s_%03d.mat', out_dir, lab, s);
        save(fname, 't', 'sig', 'fs');

        manifest{idx,1} = fname;
        manifest{idx,2} = lab;
        idx = idx + 1;
    end
end

manifest_tbl = cell2table(manifest, 'VariableNames', {'file','label'});
writetable(manifest_tbl, '../data/manifest.csv');

fprintf("✅ Generated %d signals and manifest.csv\n", size(manifest,1));
