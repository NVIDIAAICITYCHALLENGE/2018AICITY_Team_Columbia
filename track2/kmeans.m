%for accident

% 
% run('/home/wei/vlfeat/toolbox/vl_setup');
% data = zeros(627600,256);
% 
% row_count = 1;
% 
% path = '/data/LSDE/CV/wei/all_data_pca/ACCIDENT/';
% 
% fprintf('accident');
% 
% for i=1:277
%     fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
%     
%     fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
% end
% 
% fprintf('HAZARD_CAR_STOPPED');
% path = '/data/LSDE/CV/wei/all_data_pca/HAZARD_CAR_STOPPED/';
% 
% for i=1:1970
%     fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
%     
%     fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
% end
% 
% path = '/data/LSDE/CV/wei/all_data_pca/CONSTRUCTION/';
% 
% fprintf('CONSTRUCTION');
% 
% for i=1:2404
%     fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
%     
%     fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
% end
% 
% path = '/data/LSDE/CV/wei/all_data_pca/POLICE/';
% fprintf('POLICE');
% 
% 
% for i=1:416
%     fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
%     
%     fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
% end
% 
% path = '/data/LSDE/CV/wei/all_data_pca/ROAD_CLOSED/';
% fprintf('ROAD_CLOSED');
% 
% for i=1:610
%     fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
%     
%     fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
% end
% 
% 
% path = '/data/LSDE/CV/wei/all_data_pca/PARTIAL_CLOSURE/';
% fprintf('PARTIAL_CLOSURE');
% 
% for i=1:599
%     fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
%     
%     fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
%     data_chunk = csvread(fullpath);
%     num_row = size(data_chunk,1);
%     data(row_count:row_count+num_row-1,:) = data_chunk;
%     row_count = row_count + num_row;
% end
% 
% data(all(data==0,2),:) = [];
% 
% data = data';
% 
% centers = vl_kmeans(data, 7);
% 
% csvwrite('centers.csv',centers);

numFeatures = 5000 ;
dimension = 2 ;
data = rand(dimension,numFeatures) ;
dataEncode = rand(dimension,1000);
numClusters = 30 ;
centers = vl_kmeans(data, numClusters);
kdtree = vl_kdtreebuild(centers) ;
nn = vl_kdtreequery(kdtree, centers, dataEncode) ;
assignments = zeros(numClusters,1000);
assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
enc = vl_vlad(dataEncode,centers,assignments,'NormalizeComponents');