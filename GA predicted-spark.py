import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
color = sns.color_palette()

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as func
spark=SparkSession.builder.appName('my_first_app_name').getOrCreate()

train_df = spark.read.csv(SCRIPT_PATH +'/data/train_processed.csv', header=True, inferSchema=True)
test_df = spark.read.csv(SCRIPT_PATH +'/data/test_processed.csv', header=True, inferSchema=True)

for col in train_df.columns:
    c = col.replace(".","")
    train_df = train_df.withColumnRenamed(col,c)
for col in test_df.columns:
    c = col.replace(".","")
    test_df = test_df.withColumnRenamed(col,c)

# print('train_df = dtpyes',train_df.dtypes)
# print('test_df = dtpyes',test_df.dtypes)

train_df = train_df.withColumn("totalstransactionRevenue", train_df["totalstransactionRevenue"].cast('float'))
gdf = train_df.groupby("fullVisitorId").agg(func.sum("totalstransactionRevenue"))

# cols =  ['socialEngagementType',
#         'devicebrowserSize',
#         'devicebrowserVersion',
#         'deviceflashVersion',
#         'devicelanguage',
#         'devicemobileDeviceBranding',
#         'devicemobileDeviceInfo',
#         'devicemobileDeviceMarketingName',
#         'devicemobileDeviceModel',
#         'devicemobileInputSelector',
#         'deviceoperatingSystemVersion',
#         'devicescreenColors',
#         'devicescreenResolution',
#         'geoNetworkcityId',
#         'geoNetworklatitude',
#         'geoNetworklongitude', 
#         'geoNetworknetworkLocation',
#         'totalsvisits',
#         'trafficSourceadwordsClickInfocriteriaParameters']

cols = [c for c in train_df.columns if train_df.select(c).distinct().count()==1]
cols_to_drop = cols + ['sessionId'] + ["trafficSourcecampaignCode"]
for c in cols_to_drop:
    train_df = train_df.drop(c)
    test_df = test_df.drop(c)

# # Impute 0 for missing target values
train_df = train_df.fillna(0, subset=['totalstransactionRevenue'])
train_y = train_df.select(['totalstransactionRevenue'])
train_id = train_df.select(['fullVisitorId'])
test_id = test_df.select(['fullVisitorId'])

from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import udf, log1p
from pyspark.sql.types import StringType

cat_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent",
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']

# print(train_df.select(['trafficSourceadContent']).show())

# some column cannot compute
cats = []
for col in cat_cols:
    c = col.replace(".","")
    udf1 = udf(lambda x: x if x is not None else "None",StringType())
    test_df = test_df.withColumn(c,udf1(c))
    train_df = train_df.withColumn(c,udf1(c))
    indexer = StringIndexer(inputCol=c, outputCol=c+"_lbl")
    train_df = indexer.fit(train_df).transform(train_df)
    indexer = StringIndexer(inputCol=c, outputCol=c+"_lbl")
    test_df = indexer.fit(test_df).transform(test_df)
    train_df = train_df.drop(c)
    test_df = test_df.drop(c)
    cats.append(c+"_lbl")

print('111')

nums = []
num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
for col in num_cols:
    col = col.replace(".","")
    train_df = train_df.withColumn(col, train_df[col].cast('float'))
    test_df = test_df.withColumn(col, test_df[col].cast('float'))
    nums.append(col)

from pyspark.sql.functions import unix_timestamp, lit
from pyspark.sql.types import DateType

udf1 = udf(lambda x:x[0:4]+'-'+x[4:6]+'-'+x[6:],StringType())
train_df = (train_df.withColumn("date", train_df["date"].cast("string"))).withColumn('date',udf1('date'))
train_df = train_df.withColumn("date",train_df['date'].cast(DateType()))

dev_df = train_df.filter(train_df["date"] <= lit('2017-03-01'))
val_df = train_df.filter(train_df["date"] > lit('2017-03-01'))

print('dev_df = dtpyes',dev_df.dtypes)
print('val_df = dtpyes',val_df.dtypes)

dev_y = dev_df.withColumn("totalstransactionRevenuelog1p",log1p('totalstransactionRevenue')).select(['totalstransactionRevenuelog1p'])
val_y = val_df.withColumn("totalstransactionRevenuelog1p",log1p('totalstransactionRevenue')).select(['totalstransactionRevenuelog1p'])

dev_df = dev_df.toPandas()
val_df = val_df.toPandas()
dev_y = dev_y.toPandas()
val_y = val_y.toPandas()
test_df = test_df.toPandas()

dev_X = dev_df[cats + nums] 
val_X = val_df[cats + nums] 
test_X = test_df[cats + nums] 

# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

from sklearn import metrics
pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totalstransactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))

sub_df = pd.DataFrame({"fullVisitorId":test_df["fullVisitorId"].values})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv(SCRIPT_PATH +'/data/GA_version_1_lgb.csv', index=False)

print(sub_df.head())
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
