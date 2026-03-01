% path to subject

sub_path = "/Users/omriraccah/Documents/Projects_Postdoc/Multisensory_Memory/fMRI_expt_multisensory_memory/subjects/mm26/";

file_name = "events_run_9_sub_mm26_unfixed.mat";

load(append(sub_path,file_name))

data_new = struct;

%% save data.durations and data.onsets: all stim and ITI durations
counter = 1;
for t = 1:40
    data.Onsets(counter) = data.stimStart(t);
    data.Durations(counter) = data.stimEnd(t)- data.stimStart(t);
    counter = counter + 1;
    data.Onsets(counter) = data.ITIStart(t);
    data.Durations(counter) = data.ITIEnd(t) - data.ITIStart(t);
    counter = counter + 1;
end

%% save the data
save("events_run_9_sub_mm26",'data','xp');
