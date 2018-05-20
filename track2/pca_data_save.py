import psycopg2
import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA
import csv
import datetime as dt
import io
from sklearn.externals import joblib



try:
    conn = psycopg2.connect("dbname='larsde_other' user='flask' host='larsde.cs.columbia.edu' password='dvmm32123'")
    conn.autocommit= True
except Exception, e:
    print e, "Connection Unsucessful"''

cur = conn.cursor()

video_seg = []
with open('untrim_data/untrim_seg.csv', 'rU') as f:
    reader = csv.reader(f)
    reader = csv.reader(f, delimiter=',', quotechar='|')
    video_seg = list(reader)

event_seg = []
with open('untrim_data/event_seg.csv', 'rU') as f:
    reader = csv.reader(f)
    reader = csv.reader(f)
    event_seg = list(reader)


#pca = PCA(n_components=256)


for idx, seg in enumerate(video_seg):
    all_data = np.zeros((1, 4096))
    cam_id = int(seg[0])
    start_time = dt.datetime.strptime(seg[1], '%m/%d/%Y %H:%M:%S')
    end_time = dt.datetime.strptime(seg[2], '%m/%d/%Y %H:%M:%S')

    cur.execute("SELECT content FROM image_features where new_id = %d and time >= \'%s\' and time <= \'%s\' limit 5000 " % (cam_id, start_time, end_time, ))

    counter = 0
    for feat in cur:
        print "this is the feature number:" + str(counter) + " event number: " + str(idx)
        counter += 1
        if feat[0] is None:
            continue
        buf = io.BytesIO()
        buf.write(feat[0])
        buf.seek(0)
        featarr = np.load(buf)
        featarr = featarr.reshape((1, 4096))
        all_data = np.concatenate((all_data, featarr), axis=0)

    all_data = np.delete(all_data, 0, axis=0)
    name = "alldata/" + str(idx) + ".csv"
    np.savetxt(name,all_data,delimiter=",")



cur.close()
conn.close()