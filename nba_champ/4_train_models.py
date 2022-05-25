# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

# Part 4: Model Training

# This script is used to train an Explained model using the Jobs feature
# in CML and the Experiments feature to facilitate model tuning

# If you haven't yet, run through the initialization steps in the README file and Part 1.
# In Part 1, the data is imported into the table you specified in Hive.
# All data accesses fetch from Hive.
#
# To simply train the model once, run this file in a workbench session.
#
# There are 2 other ways of running the model training process
#
# ***Scheduled Jobs***
#
# The **[Jobs](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html)**
# feature allows for adhoc, recurring and depend jobs to run specific scripts. To run this model
# training process as a job, create a new job by going to the Project window and clicking _Jobs >
# New Job_ and entering the following settings:
# * **Name** : Train Mdoel
# * **Script** : 4_train_models.py
# * **Arguments** : _Leave blank_
# * **Kernel** : Python 3
# * **Schedule** : Manual
# * **Engine Profile** : 1 vCPU / 2 GiB
# The rest can be left as is. Once the job has been created, click **Run** to start a manual
# run for that job.

# ***Experiments***
#
# Training a model for use in production requires testing many combinations of model parameters
# and picking the best one based on one or more metrics.
# In order to do this in a *principled*, *reproducible* way, an Experiment executes model training code with **versioning** of the **project code**, **input parameters**, and **output artifacts**.
# This is a very useful feature for testing a large number of hyperparameters in parallel on elastic cloud resources.

# **[Experiments](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-running-an-experiment.html)**.
# run immediately and are used for testing different parameters in a model training process.
# In this instance it would be use for hyperparameter optimisation. To run an experiment, from the
# Project window click Experiments > Run Experiment with the following settings.
# * **Script** : 4_train_models.py
# * **Arguments** : 5 lbfgs 100 _(these the cv, solver and max_iter parameters to be passed to
# LogisticRegressionCV() function)
# * **Kernel** : Python 3
# * **Engine Profile** : 1 vCPU / 2 GiB

# Click **Start Run** and the expriment will be sheduled to build and run. Once the Run is
# completed you can view the outputs that are tracked with the experiment using the
# `cdsw.track_metrics` function. It's worth reading through the code to get a sense of what
# all is going on.

# More Details on Running Experiments
# Requirements
# Experiments have a few requirements:
# - model training code in a `.py` script, not a notebook
# - `requirements.txt` file listing package dependencies
# - a `cdsw-build.sh` script containing code to install all dependencies
#
# These three components are provided for the churn model as `4_train_models.py`, `requirements.txt`,
# and `cdsw-build.sh`, respectively.
# You can see that `cdsw-build.sh` simply installs packages from `requirements.txt`.
# The code in `4_train_models.py` is largely identical to the code in the last notebook.
# with a few differences.
#
# The first difference from the last notebook is at the "Experiments options" section.
# When you set up a new Experiment, you can enter
# [**command line arguments**](https://docs.python.org/3/library/sys.html#sys.argv)
# in standard Python fashion.
# This will be where you enter the combination of model hyperparameters that you wish to test.
#
# The other difference is at the end of the script.
# Here, the `cdsw` package (available by default) provides
# [two methods](https://docs.cloudera.com/machine-learning/cloud/experiments/topics/ml-tracking-metrics.html)
# to let the user evaluate results.
#
# **`cdsw.track_metric`** stores a single value which can be viewed in the Experiments UI.
# Here we store two metrics and the filepath to the saved model.
#
# **`cdsw.track_file`** stores a file for later inspection.
# Here we store the saved model, but we could also have saved a report csv, plot, or any other
# output file.
#


#from pyspark.sql.types import *
#from pyspark.sql import SparkSession
#import sys
import os
import pandas as pd
import numpy as np
#import cdsw
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.compose import ColumnTransformer
from lime.lime_tabular import LimeTabularExplainer
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


try:
  os.chdir("code")
except:
  pass
from churnexplainer import ExplainedModel, CategoricalEncoder

# bring in nba data set that was saved as csv
df = pd.read_csv('nba.csv')

# drop repeat season/year column
df.drop('Year', axis=1, inplace=True)

# drop name, since we already have abbreviation
df.drop('name', axis=1, inplace=True)

# convert Champion column to int with a 1 if team won championship that year, and 0 if not
df['Champion'] = df['Champion'].replace('.*', 'Yes', regex=True)
df['Champion'] = df['Champion'].fillna('No')

# create a unique id
df['uid'] = df['abbreviation'] + '_' + df['season'].astype('int').astype('str')

# drop abbreviation to make my life easier
df = df.drop('abbreviation', axis=1)

# scale everything to the number of games played in the season
cols = df.columns.to_list()
do_not_scale = ['uid', 'season', 'Champion', 'games_played', 'rank'] + [col for col in cols if 'percentage' in col]
do_scale = [col for col in cols if col not in do_not_scale]
scaled_df = df[do_scale].div(df['games_played'], axis=0)
df = pd.concat([df[do_not_scale], scaled_df], axis=1)

# save column names to be used later
idcol = 'uid'  # ID column
labelcol = 'Champion'  # label column
cols = ['rank',
        'assists',
        'blocks',
        'defensive_rebounds',
        'field_goal_attempts',
        'field_goal_percentage',
        'free_throw_attempts',
        'free_throw_percentage',
        'offensive_rebounds',
        'opp_assists',
        'opp_blocks',
        'opp_defensive_rebounds',
        'opp_field_goal_attempts',
        'opp_field_goal_percentage',
        'opp_free_throw_attempts',
        'opp_free_throw_percentage',
        'opp_offensive_rebounds',
        'opp_personal_fouls',
        'opp_points',
        'opp_steals',
        'opp_three_point_field_goal_attempts',
        'opp_three_point_field_goal_percentage',
        'opp_total_rebounds',
        'opp_turnovers',
        'opp_two_point_field_goal_attempts',
        'opp_two_point_field_goal_percentage',
        'personal_fouls',
        'points',
        'steals',
        'three_point_field_goal_attempts',
        'three_point_field_goal_percentage',
        'total_rebounds',
        'turnovers',
        'two_point_field_goal_attempts',
        'two_point_field_goal_percentage',
        'season']

# pulling out 2022 data as that is what we'll be predicting
df_2022 = df[df['season'] == 2022]
#df_2022.to_csv('nba_2022.csv', index=False)
df = df[df['season'] < 2022]

# Train on full data set
X = df[cols]
y = df[labelcol]
pipe = Pipeline(steps=[("sampling", SMOTE(sampling_strategy=.25)),
                       ("scaler", StandardScaler()),
                       ("pca", PCA(n_components=5)),
                       ("clf", SVC(C=1,
                                   class_weight='balanced',
                                   kernel='rbf',
                                   gamma=.001,
                                   probability=True))])
pipe.fit(X, y)

# Calculate probability for all 2022 data points
data = df_2022.drop(labelcol, axis=1)
data["Champion probability"] = pipe.predict_proba(data[cols])[:, 1]
data = data[['uid'] + cols + ['Champion probability']]
data[labelcol] = np.nan
data = data.sort_values('Champion probability', ascending=False)

# Create LIME Explainer
class_names = ["No " + df[labelcol].name, df[labelcol].name]
explainer = LimeTabularExplainer(
    df[cols].to_numpy(),
    feature_names=cols,
    class_names=class_names,
    mode='classification' 
)

# Create and save the combined Logistic Regression and LIME Explained Model.
explainedmodel = ExplainedModel(
    data=data,
    labels=data[labelcol], #labels=df[labelcol],
    pipeline=pipe,
    explainer=explainer,
    features=cols
)
explainedmodel.save(model_name='telco_linear')

# Wrap up

# We've now covered all the steps to **running Experiments**.
#
# Notice also that any script that will run as an Experiment can also be run as a Job or in a Session.
# Our provided script can be run with the same settings as for Experiments.
# A common use case is to **automate periodic model updates**.
# Jobs can be scheduled to run the same model training script once a week using the latest data.
# Another Job dependent on the first one can update the model parameters being used in production
# if model metrics are favorable.
