def preprocess(filename):
  import csv
  import numpy as np
  f=open(filename,"r")
  reader=csv.reader(f)
  featurename=[]
  allfeatures=[]
  alltargets=[]
  for i,line in enumerate(reader):
    if i==0:
      featurename=line[:-1]
    else:
      allfeatures.append(line[2:])
      alltargets.append(line[1])
  alltargets=list(map(lambda x:x=="M",alltargets))
  f.close()
  return featurename,np.array(allfeatures),np.array(alltargets).reshape((len(alltargets),1))
