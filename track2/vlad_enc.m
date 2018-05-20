run('/home/wei/vlfeat/toolbox/vl_setup');

row_count = 1;

centers = csvread('centers.csv');

path = '/data/LSDE/CV/wei/all_data_pca/ACCIDENT/';

fprintf('accident');

numClusters = 7;

for i=1:277
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
    
    
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
end

fprintf('HAZARD_CAR_STOPPED');
path = '/data/LSDE/CV/wei/all_data_pca/HAZARD_CAR_STOPPED/';

for i=1:1970
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
    
    
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
end

path = '/data/LSDE/CV/wei/all_data_pca/CONSTRUCTION/';

fprintf('CONSTRUCTION');

for i=1:2404
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
    
    
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
end

path = '/data/LSDE/CV/wei/all_data_pca/POLICE/';
fprintf('POLICE');


for i=1:416
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
    
    
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
end

path = '/data/LSDE/CV/wei/all_data_pca/ROAD_CLOSED/';
fprintf('ROAD_CLOSED');

for i=1:610
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
    
    
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
end


path = '/data/LSDE/CV/wei/all_data_pca/PARTIAL_CLOSURE/';
fprintf('PARTIAL_CLOSURE');

for i=1:599
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'event/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
    
    
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'.csv');
    data_chunk = csvread(fullpath);
    numDataToBeEncoded = size(data_chunk,1);
    kdtree = vl_kdtreebuild(centers) ;
    nn = vl_kdtreequery(kdtree, centers, data_chunk') ;
    assignments = zeros(numClusters,numDataToBeEncoded);
    assignments(sub2ind(size(assignments), nn, 1:length(nn))) = 1;
    enc = vl_vlad(data_chunk',centers,assignments,'NormalizeComponents');
    fullpath = strcat(path,'background/',num2str(i-1),'/',num2str(i-1),'_vlad','.csv');
    csvwrite(fullpath,enc);
end
