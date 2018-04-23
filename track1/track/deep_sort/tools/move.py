import os
'''
path = '../DETRAC-train-data'
for dir in os.listdir(path):
    dir_path = os.path.join(path,dir)
    save_path = os.path.join(dir_path,'img1')
    for f in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path,f)):
            os.rename(os.path.join(dir_path,f),os.path.join(save_path,f))
            #print(os.path.join(save_path,f))
    print('Finished moving to', save_path)
'''


path = '../DETRAC-train-data'
detect_path = '../detrac_train_detect_txt/train_detect_txt'

for dir in os.listdir(path):
    dir_path = os.path.join(path,dir)
    save_path = os.path.join(dir_path,'det')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    detect_file = os.path.join(detect_path,dir+'_det.txt')
    if os.path.exists(os.path.join(save_path,dir+'_det.txt')):
            os.remove(os.path.join(save_path,dir+'_det.txt'))
    os.rename(detect_file,os.path.join(save_path,dir+'_det.txt'))
    


