# Databricks notebook source
# MAGIC %md # Training machine learning models on tabular data: an end-to-end example
# MAGIC 
# MAGIC This tutorial covers the following steps:
# MAGIC - Import data from your local machine into the Databricks File System (DBFS)
# MAGIC - Visualize the data using Seaborn and matplotlib
# MAGIC - Run a parallel hyperparameter sweep to train machine learning models on the dataset
# MAGIC - Explore the results of the hyperparameter sweep with MLflow
# MAGIC - Register the best performing model in MLflow
# MAGIC - Apply the registered model to another dataset using a Spark UDF
# MAGIC - Set up model serving for low-latency requests
# MAGIC 
# MAGIC In this example, you build a model to predict the quality of Portugese "Vinho Verde" wine based on the wine's physicochemical properties. 
# MAGIC 
# MAGIC The example uses a dataset from the UCI Machine Learning Repository, presented in [*Modeling wine preferences by data mining from physicochemical properties*](https://www.sciencedirect.com/science/article/pii/S0167923609001377?via%3Dihub) [Cortez et al., 2009].
# MAGIC 
# MAGIC ## Requirements
# MAGIC This notebook requires Databricks Runtime for Machine Learning.  
# MAGIC If you are using Databricks Runtime 7.3 LTS ML or below, you must update the CloudPickle library. To do that, uncomment and run the `%pip install` command in Cmd 2. 

# COMMAND ----------

#/Repos/megan.masanz@microsoft.com/Azure-Databricks-MLOps/notebooks/MyProject/StoneX ML End-to-End Example

# COMMAND ----------

# This command is only required if you are using a cluster running DBR 7.3 LTS ML or below. 
#%pip install --upgrade cloudpickle

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC   
# MAGIC In this section, you download a dataset from the web and upload it to Databricks File System (DBFS).
# MAGIC 
# MAGIC 1. Navigate to https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/ and download both `winequality-red.csv` and `winequality-white.csv` to your local machine.
# MAGIC 
# MAGIC 1. From this Databricks notebook, select **File > Upload data to DBFS...**, and drag these files to the drag-and-drop target to upload them to the Databricks File System (DBFS). 
# MAGIC 
# MAGIC     **Note**: if you don't have the **File > Upload data to DBFS...** option, you can load the dataset from the Databricks example datasets. Uncomment and run the last two lines in the following cell.
# MAGIC 
# MAGIC 1. Click **Next**. Auto-generated code to load the data appears. Under **Access Files from Notebooks**, select the **pandas** tab. Click **Copy** to copy the example code, and then click **Done**. 
# MAGIC 
# MAGIC 1. Create a new cell, then paste in the sample code. It will look similar to the code shown in the following cell. Make these changes:
# MAGIC   - Pass `sep=';'` to `pd.read_csv`
# MAGIC   - Change the variable names from `df1` and `df2` to `white_wine` and `red_wine`, as shown in the following cell.

# COMMAND ----------

import pandas as pd

white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=";")
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=";")

# COMMAND ----------

# MAGIC %md Merge the two DataFrames into a single dataset, with a new binary feature "is_red" that indicates whether the wine is red or white.

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0

data = pd.concat([red_wine, white_wine], axis=0)

# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

# COMMAND ----------

data.head()

# COMMAND ----------

# MAGIC %md ## Visualize data
# MAGIC 
# MAGIC Before training a model, explore the dataset using Seaborn and Matplotlib.

# COMMAND ----------

# MAGIC %md Plot a histogram of the dependent variable, quality.

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# MAGIC %md Looks like quality scores are normally distributed between 3 and 9. 
# MAGIC 
# MAGIC Define a wine as high quality if it has quality >= 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

# MAGIC %md Box plots are useful in noticing correlations between features and a binary label.

# COMMAND ----------

import matplotlib.pyplot as plt

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0

# COMMAND ----------

# MAGIC %md In the above box plots, a few variables stand out as good univariate predictors of quality. 
# MAGIC 
# MAGIC - In the alcohol box plot, the median alcohol content of high quality wines is greater than even the 75th quantile of low quality wines. High alcohol content is correlated with quality.
# MAGIC - In the density box plot, low quality wines have a greater density than high quality wines. Density is inversely correlated with quality.

# COMMAND ----------

# MAGIC %md ## Preprocess data
# MAGIC Prior to training a model, check for missing values and split the data into training and validation sets.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md There are no missing values.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare dataset for training baseline model
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = data.drop(["quality"], axis=1)
y = data.quality

# Split out the training data
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)

# Split the remaining data equally into validation and test
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md ## Build a baseline model
# MAGIC This task seems well suited to a random forest classifier, since the output is binary and there may be interactions between multiple variables.
# MAGIC 
# MAGIC The following code builds a simple classifier using scikit-learn. It uses MLflow to keep track of the model accuracy, and to save the model for later use.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time

#mlflow.set_experiment(experiment_name = '/Repos/megan.masanz@microsoft.com/Azure-Databricks-MLOps/notebooks/MyProject/StoneX ML End-to-End Example')

mlflow.set_experiment(experiment_name =	'/MyProject/StoneX ML End-to-End Example')

# COMMAND ----------



# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]

# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)

  # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use the area under the ROC curve as a metric.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # Log the model with a signature that defines the schema of the model's inputs and outputs. 
  # When the model is deployed, this signature will be used to validate inputs.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflow contains utilities to create a conda environment used to serve models.
  # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__), "mlflow<=1.30.0"],
        additional_conda_channels=None,
    )
  model_info = mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)
  print(model_info)

# COMMAND ----------

# MAGIC %md Examine the learned feature importances output by the model as a sanity-check.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md As illustrated by the boxplots shown previously, both alcohol and density are important in predicting quality.

# COMMAND ----------

# MAGIC %md You logged the Area Under the ROC Curve (AUC) to MLflow. Click **Experiment** at the upper right to display the Experiment Runs sidebar. 
# MAGIC 
# MAGIC The model achieved an AUC of 0.854.
# MAGIC 
# MAGIC A random classifier would have an AUC of 0.5, and higher AUC values are better. For more information, see [Receiver Operating Characteristic Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).

# COMMAND ----------

# MAGIC %md #### Register the model in MLflow Model Registry
# MAGIC 
# MAGIC By registering this model in Model Registry, you can easily reference the model from anywhere within Databricks.
# MAGIC 
# MAGIC The following section shows how to do this programmatically, but you can also register a model using the UI. See "Create or register a model using the UI" ([AWS](https://docs.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/manage-model-lifecycle/index#create-or-register-a-model-using-the-ui)|[GCP](https://docs.gcp.databricks.com/applications/machine-learning/manage-model-lifecycle/index.html#create-or-register-a-model-using-the-ui)).

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id
print('run_id = ' + run_id)

client = mlflow.tracking.MlflowClient()
client.list_artifacts(run_id=run_id)

# COMMAND ----------

import os
local_dir = "/dbfs/FileStore/temp/" + run_id
if not os.path.exists(local_dir):
  os.mkdir(local_dir)

 # Download the artifact to local storage.
local_path = client.download_artifacts(run_id, "random_forest_model", local_dir)
print("Artifacts downloaded in: {}".format(local_dir))
print("Artifacts: {}".format(local_dir))

# COMMAND ----------

dbutils.fs.ls('dbfs:/FileStore/temp/' + run_id + "/random_forest_model")

# COMMAND ----------

model_path = 'dbfs:/FileStore/temp/' + run_id + "/random_forest_model"

# COMMAND ----------

# from dotenv import load_dotenv
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

import os
os.environ['AZURE_CLIENT_ID'] = "66c790b4-6716-4b0b-86f4-96f13e826da2"
os.environ['AZURE_TENANT_ID'] = "16b3c013-d300-468d-ac64-7eda0820b6d3"
os.environ['AZURE_CLIENT_SECRET'] = "7aA8Q~fghrXUAeDYkLIrNLqfDl3zfH-fB5Nhhatl"

subscription_id =  "b071bca8-0055-43f9-9ff8-ca9a144c2a6f"
resource_group =  "aml-dev-main-rg"
workspace_name =  "aml-dev-main"


credential = DefaultAzureCredential()
# Check if given credential can get token successfully.
credential.get_token("https://management.azure.com/.default")
#ml_client = MLClient.from_config(credential, subscription_id, resource_group, workspace_name)
ml_client = MLClient(
    credential, subscription_id, resource_group, workspace_name
)
azureml_mlflow_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
print(azureml_mlflow_uri)
mlflow.set_registry_uri(azureml_mlflow_uri)

# COMMAND ----------

model_path = '/dbfs/FileStore/temp/' + run_id + "/random_forest_model"

# COMMAND ----------

model_info.model_uri

# COMMAND ----------

# If you see the error "PERMISSION_DENIED: User does not have any permission level assigned to the registered model", 
# the cause may be that a model already exists with the name "wine_quality". Try using a different name.
model_name = "wine_quality"

from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

#'dbfs:/FileStore/temp/' + run_id + "/random_forest_model"
# run_model = Model(
#     path=f"runs:/{last_run.info.run_id}/model",
#     name="titanic_model",
#     description="Model created from run.",
#     type=AssetTypes.MLFLOW_MODEL 
# )

run_model = Model(
    path=model_path,
    name="titanic_model",
    description="Model created from run.",
    type=AssetTypes.MLFLOW_MODEL,
    tags = [['model_uri:', model_info.model_uri ]]
)

model = ml_client.models.create_or_update(run_model)

# COMMAND ----------

# from azure.ai.ml.entities import (
#     ManagedOnlineEndpoint,
#     ManagedOnlineDeployment,
#     Model,
#     Environment,
#     CodeConfiguration,
# )

# online_endpoint_name = "mmstonexendpoint"
# endpoint = ManagedOnlineEndpoint(
#     name=online_endpoint_name,
#     description="wine online endpoint for mlflow model from databricks",
#     auth_mode="key",
#     tags={"mlflow": "true"},
# )

# COMMAND ----------

# ml_client.begin_create_or_update(endpoint)

# COMMAND ----------

# MAGIC %md You should now see the model in the Models page. To display the Models page, click the Models icon in the left sidebar. 
# MAGIC 
# MAGIC Next, transition this model to production and load it into this notebook from Model Registry.

# COMMAND ----------

# MAGIC %md The Models page now shows the model version in stage "Production".
# MAGIC 
# MAGIC You can now refer to the model using the path "models:/wine_quality/production".

# COMMAND ----------

X_test

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_info.model_uri)

# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

import json

request_data =  {
              "input_data": {
                "columns": [
                  "fixed_acidity",
                  "volatile_acidity",
                  "citric_acid",
                  "residual_sugar",
                  "chlorides",
                  "free_sulfur_dioxide",
                  "total_sulfur_dioxide",
                  "density",
                  "pH",
                  "sulphates",
                  "alcohol",
                  "is_red"
                ],
                "data": []
              }
            }

request_df = X_test.head(2)
request_data['input_data']['data'] = json.loads(request_df.to_json(orient='records'))
parsed = json.dumps(request_data)
print(parsed)

# COMMAND ----------


