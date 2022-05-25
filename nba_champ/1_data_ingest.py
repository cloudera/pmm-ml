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

# Part 1: Data Ingest
# A data scientist should never be blocked in getting data into their environment,
# so CML is able to ingest data from many sources.
# Whether you have data in .csv files, modern formats like parquet or feather,
# in cloud storage or a SQL database, CML will let you work with it in a data
# scientist-friendly environment.

# Access local data on your computer
#
# Accessing data stored on your computer is a matter of [uploading a file to the CML filesystem and
# referencing from there](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-accessing-local-data-from-your-computer.html).
#
# > Go to the project's **Overview** page. Under the **Files** section, click **Upload**, select the relevant data files to be uploaded and a destination folder.
#
# If, for example, you upload a file called, `mydata.csv` to a folder called `data`, the
# following example code would work.

# ```
# import pandas as pd
#
# df = pd.read_csv('data/mydata.csv')
#
# # Or:
# df = pd.read_csv('/home/cdsw/data/mydata.csv')
# ```

# Access data in S3
#
# Accessing [data in Amazon S3](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-accessing-data-in-amazon-s3-buckets.html)
# follows a familiar procedure of fetching and storing in the CML filesystem.
# > Add your Amazon Web Services access keys to your project's
# > [environment variables](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-environment-variables.html)
# > as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.
#
# To get the the access keys that are used for you in the CDP DataLake, you can follow
# [this Cloudera Community Tutorial](https://community.cloudera.com/t5/Community-Articles/How-to-get-AWS-access-keys-via-IDBroker-in-CDP/ta-p/295485)

#
# The following sample code would fetch a file called `myfile.csv` from the S3 bucket, `data_bucket`, and store it in the CML home folder.
# ```
# # Create the Boto S3 connection object.
# from boto.s3.connection import S3Connection
# aws_connection = S3Connection()
#
# # Download the dataset to file 'myfile.csv'.
# bucket = aws_connection.get_bucket('data_bucket')
# key = bucket.get_key('myfile.csv')
# key.get_contents_to_filename('/home/cdsw/myfile.csv')
# ```


# Access data from Cloud Storage or the Hive metastore
#
# Accessing data from [the Hive metastore](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-accessing-data-from-apache-hive.html)
# that comes with CML only takes a few more steps.
# But first we need to fetch the data from Cloud Storage and save it as a Hive table.
#
# > First we specify `STORAGE` as an
# > [environment variable](https://docs.cloudera.com/machine-learning/cloud/import-data/topics/ml-environment-variables.html)
# > in your project settings containing the Cloud Storage location used by the DataLake to store
# > Hive data. On AWS it will be `s3a://[something]`, on Azure it will be `abfs://[something]` and on
# > on prem CDSW cluster, it will be `hdfs://[something]`
#
# This was done for you when you ran `0_bootstrap.py`, so the following code is set up to run as is.
# It begins with imports and creating a `SparkSession`.

# for issue with sportsipy ->
# https://stackoverflow.com/questions/70519889/when-i-run-the-sportsipy-nba-teams-teams-function-i-am-getting-notified-that-the

import pandas as pd
from sportsipy.nba.teams import Teams
from datetime import datetime
from requests import get
from bs4 import BeautifulSoup

# pull individual team data for each season for the 3 point era
team_stats_list = []
#iterate through each season
for year in range(1980,2023):
    #iterate through each team for that season
    for team in Teams(year):
        temp_df = team.dataframe
        temp_df['season'] = year
        team_stats_list.append(temp_df)
nba_df = pd.concat(team_stats_list).reset_index(drop=True)

# scrape champion for each year from basketball-reference.com
r = get(f'https://www.basketball-reference.com/playoffs/')
if r.status_code==200:
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find('table', attrs={'id': 'champions_index'})
    if table:
        champ_df = pd.read_html(str(table))[0]
        # frop top level of multi index
        champ_df = champ_df.droplevel(level=0, axis=1)
        # filter to only NBA champions
        champ_df = champ_df[champ_df['Lg'] == 'NBA']
        # filter to only champions since 3 point era
        champ_df = champ_df[champ_df['Year'] > 1979]
        # only keep Year, Champion
        champ_df = champ_df[['Year', 'Champion']]
        # change year format to int
        champ_df['Year'] = champ_df['Year'].astype(int)

# merge df's based on team name and season
full_df = pd.merge(nba_df,
                   champ_df,
                   left_on=['name', 'season'],
                   right_on=['Champion', 'Year'],
                   how='outer')

full_df = full_df[full_df['abbreviation'].notna()]

full_df.to_csv('code/nba.csv', index=False)

# Other ways to access data

# To access data from other locations, refer to the
# [CML documentation](https://docs.cloudera.com/machine-learning/cloud/import-data/index.html).

# Scheduled Jobs
#
# One of the features of CML is the ability to schedule code to run at regular intervals,
# similar to cron jobs. This is useful for **data pipelines**, **ETL**, and **regular reporting**
# among other use cases. If new data files are created regularly, e.g. hourly log files, you could
# schedule a Job to run a data loading script with code like the above.

# > Any script [can be scheduled as a Job](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-job.html).
# > You can create a Job with specified command line arguments or environment variables.
# > Jobs can be triggered by the completion of other jobs, forming a
# > [Pipeline](https://docs.cloudera.com/machine-learning/cloud/jobs-pipelines/topics/ml-creating-a-pipeline.html)
# > You can configure the job to email individuals with an attachment, e.g. a csv report which your
# > script saves at: `/home/cdsw/job1/output.csv`.

# Try running this script `1_data_ingest.py` for use in such a Job.
