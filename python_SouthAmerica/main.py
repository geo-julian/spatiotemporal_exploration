# +
import os, subprocess, sys
#
# The main program of spatio temporal exploration
#
import Utils
if len(sys.argv)>1:
    Utils.load_config(sys.argv[1])
Utils.print_parameters()

#
# Step 1: convergence
#
import convergence 
#run the convergence script
#this will generate a bunch of Subduction Convergence Kinematics Statistics files
#by default the files are placed in ./convergence_data
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    convergence.run_it()
# -

#
# Step 2: Prepare training and predicting datasets    
# 
if os.path.isfile(Utils.get_coreg_input_dir() + '/positive_deposits.csv'):
    print(f'Using the deposit data in {Utils.get_coreg_input_dir()}.')
else:
    import gen_coreg_input_files
    gen_coreg_input_files.run()

#
# Step 3: coregistration
#
import coregistration
coregistration.run()

#
# Step 4: label data
#
import os
import pandas as pd
import Utils

positive_data = pd.read_csv(Utils.get_coreg_output_dir() + 'positive_deposits.csv')
negative_data = pd.read_csv(Utils.get_coreg_output_dir() + 'negative_deposits.csv')
candidates_data = pd.read_csv(Utils.get_coreg_output_dir() + 'deposit_candidates.csv')

feature_names = Utils.get_parameter('feature_names')

positive_features = positive_data[feature_names].dropna()
negative_features = negative_data[feature_names].dropna()
candidates_features = candidates_data[feature_names].dropna()

positive_features['label']=True
negative_features['label']=False

# +
#save the data
if not os.path.exists(Utils.get_ml_input_dir()):
    os.makedirs(Utils.get_ml_input_dir())
    
positive_features.to_csv(Utils.get_ml_input_dir() + 'positive.csv', index=False)
negative_features.to_csv(Utils.get_ml_input_dir() + 'negative.csv', index=False)
candidates_features.to_csv(Utils.get_ml_input_dir() + 'candidates.csv', index=False)

positive_data.iloc[positive_features.index].to_csv(Utils.get_ml_input_dir() + 'positive_all_columns.csv', index=False)
negative_data.iloc[negative_features.index].to_csv(Utils.get_ml_input_dir() + 'negative_all_columns.csv', index=False)
candidates_data.iloc[candidates_features.index].to_csv(Utils.get_ml_input_dir() + 'candidates_all_columns.csv', index=False)
# -

#
# Step 5: predict
#
#Import the tools for machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

#read in data for machine learning training and testing
candidates= pd.read_csv(Utils.get_ml_input_dir() + 'candidates.csv')
positive = pd.read_csv(Utils.get_ml_input_dir() + 'positive.csv')
negative = pd.read_csv(Utils.get_ml_input_dir() + 'negative.csv')

train_test_data = pd.concat([positive, negative])

labels = train_test_data.iloc[:,-1]
data = train_test_data.iloc[:,:-1]

#choose classifier
classifier = Utils.get_parameter('machine_learning_engine')
#classifier = 'RFC' #random forest
#classifier = 'SVC' #support vector Classification 

#preprocess features and split data for training and testing
data = preprocessing.scale(data)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=1)

if classifier == 'RFC':
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    #n_estimators use between 64-128 doi: 10.1007/978-3-642-31537-4_13
    clf = RandomForestClassifier(n_estimators=128, n_jobs=1,class_weight=None)
elif classifier == 'SVC':
    clf = SVC(probability=True,class_weight=None, gamma='auto')

#train the ML model
clf.fit(data, labels)

all_candidates_data = pd.read_csv(Utils.get_ml_input_dir() + 'candidates_all_columns.csv')

#display(candidates)
#print(all_candidates_data.iloc[positive.index])
test_data = preprocessing.scale(candidates)
mesh_prob=clf.predict_proba(test_data)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

all_candidates_data['prob'] = mesh_prob[:,1]
candidates_lat_lon = all_candidates_data[['lon','lat', 'age', 'prob']]

candidates_lat_lon = candidates_lat_lon.groupby(['lon','lat'])['prob'].mean()
#candidates_lat_lon.sort_values(ascending=False, inplace=True)

candidates_lat_lon = candidates_lat_lon.reset_index()

lons = candidates_lat_lon['lon']
lats = candidates_lat_lon['lat']

#plot the data    
fig = plt.figure(figsize=(8,6),dpi=150)
ax = plt.axes(projection=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator(range(-180,180,30))
gl.ylocator = mticker.FixedLocator(range(-90,90,15))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size': 7}
gl.ylabel_style = {'size': 7}

ax.stock_img()
ax.set_extent(Utils.get_region_of_interest_extent())
cb = ax.scatter(lons, lats, 50, marker='.',c=candidates_lat_lon['prob'], cmap=plt.cm.jet)

plt.title('Mesh Points Coloured by Mean Probability')
fig.colorbar(cb, shrink=0.5, label='Mean Probability')

if not os.path.exists(Utils.get_ml_output_dir()):
    os.makedirs(Utils.get_ml_output_dir())
plt.savefig(f'{Utils.get_ml_output_dir()}Mean_Probability.png')
plt.close()

# +
#assign rgb color
rgb=[]
probs = candidates_lat_lon['prob']
probs = (probs-min(probs))/(max(probs)-min(probs))
for p in probs:
    color = plt.cm.jet(p)
    tmp = f'#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}'.upper()
    rgb.append(tmp)
    
candidates_lat_lon['rgb'] = rgb
candidates_lat_lon.sort_values('prob', ascending=False, ignore_index=True).to_csv(
    f'{Utils.get_ml_output_dir()}predictions.csv', float_format='%.3f')
print(f'Done! The predictions have been saved in {Utils.get_ml_output_dir()}.')
# -


