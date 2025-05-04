% Read the entire file structure
fileInfo = h5info('trajectories.h5');

% Display the file structure to see what's inside
disp(fileInfo);

% Access specific datasets
% Based on the file header glimpsed in your data, there appear to be datasets like:
% - downsample_rate
% - branch_times
% - ts_data
% - _traj

% Example to read a dataset (adjust path based on actual structure)
ts_data = h5read('trajectories.h5', '/ts_data');
traj_data = h5read('trajectories.h5', '/xs_data');

[n_gene, n_timepoint, n_cell] = size(traj_data);
restructured_data = cell(1, n_timepoint);
for t = 1:n_timepoint
    restructured_data{t} = traj_data(:,t,:);
    restructured_data{t} = reshape(restructured_data{t}, [n_gene, n_cell]);
end
Tgrid = ts_data;
opts = struct;
struct.epsilon = 0.5;
[XX,YY,transportMap,J,A,D,WW,corNet,indw,indexp,TFflag,difs,out,opts] = GRITmodelSelect(restructured_data,Tgrid,[],[],opts);

filename = 'A.csv';
writematrix(A, filename);

% figure;
% plot(traj_data(1,:), traj_data(2,:), 'b-');
% title('Trajectory Data');
% xlabel('X position');
% ylabel('Y position');
% grid on;