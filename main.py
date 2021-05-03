import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


crop=pd.read_csv('Agriculture_dsc')

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100)


X=crop.drop("Crop",axis=1)
y=crop['Crop']


rf.fit(X,y)

pickle.dump(rf, open('cropmodel.pkl','wb'))