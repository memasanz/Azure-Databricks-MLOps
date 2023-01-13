# Databricks notebook source
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

cust_data = pd.read_csv('/dbfs/FileStore/datasets/clients.csv', header = 'infer')

cust_data.head()

# COMMAND ----------

cust_data.shape

# COMMAND ----------

# MAGIC %md
# MAGIC Viewing the dataset info. "num_of_dependents" variable is in categorical form(object).Also, spouse_income should have been int type but due to some typos in the data it got coerced into object

# COMMAND ----------

cust_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC Unique values are checked

# COMMAND ----------

cust_data.nunique()

# COMMAND ----------

# MAGIC %md
# MAGIC Presence of 3+ making num_dependents object type.

# COMMAND ----------

cust_data['num_of_dependents'].unique()

# COMMAND ----------

cust_data['num_of_dependents'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Null values are checked

# COMMAND ----------

cust_data.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC Dropping all rows with null values

# COMMAND ----------

cust_data.dropna(axis = 0, how = 'any', inplace = True)

# COMMAND ----------

# MAGIC %md
# MAGIC Verifying rows removal with null values

# COMMAND ----------

sum(cust_data.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC Presence of '9.857.999.878' and '1.612.000.084' is making spouse income object

# COMMAND ----------

cust_data['spouse_income'].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC Removing rows with typos of spouse income and converting spouse income into float form.

# COMMAND ----------

cust_data = cust_data[cust_data.spouse_income != '9.857.999.878']
cust_data = cust_data[cust_data.spouse_income != '1.612.000.084']

cust_data['spouse_income'] = cust_data['spouse_income'].astype(float)

# COMMAND ----------

# MAGIC %md
# MAGIC Checking descriptive stats of continuous variables

# COMMAND ----------

cust_data.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC Plotting boxplot showing relation between Income and loan_approval_status

# COMMAND ----------

fig1, ax1 = plt.subplots()

plt.ylim(0, 20000)

sns.boxplot(x = 'loan_approval_status', y = 'income', data = cust_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Similar plots are plotted

# COMMAND ----------

fig2, ax2 = plt.subplots()

sns.barplot(x = 'loan_approval_status', y = 'loan_amount', hue = 'Gender', data = cust_data)

# COMMAND ----------

fig3, ax3 = plt.subplots()

sns.barplot(x = 'loan_approval_status', y = 'spouse_income', hue = 'credit_history', data = cust_data)

# COMMAND ----------

# MAGIC %md
# MAGIC loan_approval status category counts are checked

# COMMAND ----------

cust_data['loan_approval_status'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Converting 'property_type'(nominal with 3 categories) into one hot encoded form

# COMMAND ----------

cust_data = pd.get_dummies(cust_data, columns = ['Gender', 'property_type'], drop_first = True)

cust_data.head()

# COMMAND ----------

cust_data['num_of_dependents'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC Converting all other nominal(with two categories) and ordinal variables using label encoder

# COMMAND ----------

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

cust_data['Married']= label_encoder.fit_transform(cust_data['Married'])
cust_data['education_level']= label_encoder.fit_transform(cust_data['education_level'])
cust_data['working_status']= label_encoder.fit_transform(cust_data['working_status'])
cust_data['loan_approval_status']= label_encoder.fit_transform(cust_data['loan_approval_status'])
cust_data['num_of_dependents']= label_encoder.fit_transform(cust_data['num_of_dependents'])

cust_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Verifying label encoding

# COMMAND ----------

cust_data['num_of_dependents'].value_counts()

# COMMAND ----------

cust_data.drop(labels = ['ID'], axis = 1).to_csv('/dbfs/FileStore/datasets/cust_data_processed.csv', index = False)

# COMMAND ----------

# MAGIC %md
# MAGIC Now we are splitting data into train(0.7) and test set(0.3)

# COMMAND ----------

X = cust_data.drop(labels = ['ID', 'loan_approval_status'], axis = 1)
y = cust_data['loan_approval_status']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

# COMMAND ----------

import mlflow

mlflow.set_experiment(experiment_name = '/Users/megan.masanz@microsoft.com/ADB-MLFlow-Experiment')

# COMMAND ----------

mlflow.start_run()

mlflow.log_figure(fig1, 'figure1.png')
mlflow.log_figure(fig2, 'figure2.png')
mlflow.log_figure(fig3, 'figure3.png')

mlflow.end_run()      

# COMMAND ----------

# MAGIC %md
# MAGIC We can also log run using run_name.Instead of run id, run name will be visible in run page of experiment.
# MAGIC Also, Active run id can be obtained using run.info.run_id

# COMMAND ----------

mlflow.start_run(run_name = 'exploratory_data_analysis')

mlflow.log_figure(fig1, 'boxplot_loanstatus_vs_income.png')
mlflow.log_figure(fig2, 'barplot_loanstatus_loanamt_gender.png')
mlflow.log_figure(fig3, 'barplot_spouseincome_credithistory.png')

run = mlflow.active_run()

print('Active run_id: {}'.format(run.info.run_id))

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC Alternatively we can use 'with' which automatically terminates the run at the end of the with block.We have marked the run as run1

# COMMAND ----------

with mlflow.start_run(run_name = 'eda_plots') as run1:
    mlflow.log_figure(fig1, 'boxplot_loanstatus_vs_income.png')
    mlflow.log_figure(fig2, 'barplot_loanstatus_loanamt_gender.png')
    mlflow.log_figure(fig3, 'barplot_spouseincome_credithistory.png')

      

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

with mlflow.start_run(run_name = 'RF_Default_params') as run2:
    
    model = RandomForestClassifier()

    model.fit(X_train, y_train)
    predictions =  model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision_score = precision_score(y_test, predictions)
    test_recall_score = recall_score(y_test, predictions)
    test_f1_score = f1_score(y_test, predictions)
    auc_score = roc_auc_score(y_test,  predictions_proba[:,1])
    
    mlflow.log_metric('Test_accuracy', test_accuracy )
    mlflow.log_metric('Test_precision_score', test_precision_score)
    mlflow.log_metric('Test_recall_score', test_recall_score)
    mlflow.log_metric('Test_f1_score', test_f1_score)
    mlflow.log_metric('AUC_score', auc_score)
   
    mlflow.set_tag('Classifier', 'RF-default_parameters')

# COMMAND ----------

client = mlflow.tracking.MlflowClient()

print('Model Parameters', client.get_run(run2.info.run_id).data.params)

print('Metrics', client.get_run(run2.info.run_id).data.metrics)

# COMMAND ----------

with mlflow.start_run(run_name = 'RF_tuned_params_scenario1') as run3:
  
    n_estimators = 200
    criterion = 'gini'
    min_samples_split = 5
    min_samples_leaf = 2
   
    model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion,
                                   min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    model.fit(X_train, y_train)
    predictions =  model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)
    
    mlflow.log_param('No. of trees', n_estimators)
    mlflow.log_param('Splitting criteria', criterion)
    mlflow.log_param('Min samples split', min_samples_split)
    mlflow.log_param('Min samples leaf',  min_samples_leaf)
    
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision_score = precision_score(y_test, predictions)
    test_recall_score = recall_score(y_test, predictions)
    test_f1_score = f1_score(y_test, predictions)
    auc_score = roc_auc_score(y_test,  predictions_proba[:,1])
    
    mlflow.log_metric('Test_accuracy', test_accuracy )
    mlflow.log_metric('Test_precision_score', test_precision_score)
    mlflow.log_metric('Test_recall_score', test_recall_score)
    mlflow.log_metric('Test_f1_score', test_f1_score)
    mlflow.log_metric('AUC_score', auc_score)

    mlflow.set_tag('Classifier', 'RF-tuned_params_sc1')

print('Model Parameters', client.get_run(run3.info.run_id).data.params)

print('Metrics', client.get_run(run3.info.run_id).data.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC Another two scenarios of parameters are logged.Note that both parameters and metrics can be logged using 'log_params' and 'log_metrics'.Input is in dictionary format

# COMMAND ----------

with mlflow.start_run(run_name = 'RF_tuned_params_scenario2') as run4:

    n_estimators = 200
    criterion = 'entropy'
    min_samples_split = 5
    min_samples_leaf = 2

    model = RandomForestClassifier(n_estimators = n_estimators,criterion = criterion,
                                   min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    model.fit(X_train, y_train)
    predictions =  model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision_score = precision_score(y_test, predictions)
    test_recall_score = recall_score(y_test, predictions)
    test_f1_score = f1_score(y_test, predictions)
    auc_score = roc_auc_score(y_test,  predictions_proba[:,1])
    
    params = {
        'No. of trees': n_estimators, 
        'Splitting criteria': criterion,
        'Min samples split': min_samples_split, 
        'Min samples leaf': min_samples_leaf
    }
    mlflow.log_params(params)
    
    metrics = {
        'Test_accuracy': test_accuracy, 
        'Test_precision_score': test_precision_score,
        'Test_recall_score': test_recall_score, 
        'Test_f1_score': test_f1_score, 
        'AUC_score': auc_score
    }
    mlflow.log_metrics(metrics)
    
    mlflow.set_tag('Classifier', 'RF-tuned_params_sc2')

# COMMAND ----------

with mlflow.start_run(run_name = 'RF_tuned_params_scenario3') as run5:
  
    n_estimators = 500
    criterion = 'gini'
    min_samples_split = 10
    min_samples_leaf = 4
    
    model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion,
                                 min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    model.fit(X_train, y_train)
    predictions =  model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)
    
    params = {
        'No. of trees': n_estimators, 
        'Splitting criteria': criterion,
        'Min samples split': min_samples_split, 
        'Min samples leaf': min_samples_leaf
    }
    mlflow.log_params(params)
    
    metrics = {
        'Test_accuracy': test_accuracy, 
        'Test_precision_score': test_precision_score,
        'Test_recall_score': test_recall_score, 
        'Test_f1_score': test_f1_score, 
        'AUC_score': auc_score
    }
    mlflow.log_metrics(metrics)

    mlflow.set_tag('Classifier', 'RF-tuned_params_sc3')


# COMMAND ----------

# MAGIC %md
# MAGIC Autologging parameters and metrics can also be enabled using calling mlflow.autolog or mlflow.sklearn.autolog
# MAGIC 
# MAGIC https://mlflow.org/docs/latest/tracking.html#automatic-logging. 
# MAGIC By going to the specific run page,It can be seen that all classifier parameters , training set metrics,tags,artifacts(Model,precison recall curve,ROC curve,Confusion matrix) are auto logged
# MAGIC 
# MAGIC https://learn.microsoft.com/en-gb/azure/databricks/mlflow/databricks-autologging#customize-logging-behavior

# COMMAND ----------

mlflow.sklearn.autolog()

with mlflow.start_run(run_name = 'RF_tuned_params_scenario3_autolog') as run6:
    
    n_estimators = 500
    criterion = 'gini'
    min_samples_split = 10
    min_samples_leaf = 4

    model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion,
                                   min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    model.fit(X_train, y_train)
    
    mlflow.set_tag('Classifier', 'RF-tuned_params_sc3_autolog')

# COMMAND ----------

mlflow.sklearn.autolog()

with mlflow.start_run(run_name = 'RF_tuned_params_scenario3_autolog_with_test_metrics') as run7:

    n_estimators = 500
    criterion = 'gini'
    min_samples_split = 10
    min_samples_leaf = 4
    
    model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion,
                                 min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    model.fit(X_train, y_train)
    predictions =  model.predict(X_test)
    predictions_proba = model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, predictions)
    test_precision_score = precision_score(y_test, predictions)
    test_recall_score = recall_score(y_test, predictions)
    test_f1_score = f1_score(y_test, predictions)
    
   
    mlflow.set_tag('Classifier', 'RF-tuned_params_sc3_autolog_with_test_metrics')

# COMMAND ----------

# MAGIC %md
# MAGIC List of experiments in the workspace can be obtained by this way

# COMMAND ----------

experiments_list = client.list_experiments()

experiments_list

# COMMAND ----------

# MAGIC %md
# MAGIC Experiment related details can be obtained using following method

# COMMAND ----------

experiment = mlflow.get_experiment(experiments_list[0].experiment_id)

print('Name: {}'.format(experiment.name))
print('Artifact Location: {}'.format(experiment.artifact_location))
print('Tags: {}'.format(experiment.tags))
print('Lifecycle_stage: {}'.format(experiment.lifecycle_stage))

# COMMAND ----------

# MAGIC %md
# MAGIC Alterntative way of extracting info of all experiments

# COMMAND ----------

experiment = mlflow.get_experiment_by_name('/Users/megan.masanz@microsoft.com/ADB-MLFlow-Experiment')

print('Name: {}'.format(experiment.name))
print('Artifact Location: {}'.format(experiment.artifact_location))
print('Tags: {}'.format(experiment.tags))
print('Lifecycle_stage: {}'.format(experiment.lifecycle_stage))

# COMMAND ----------

# MAGIC %md
# MAGIC Tracking uri is obtained(databricks)

# COMMAND ----------

mlflow.get_tracking_uri()

# COMMAND ----------

# MAGIC %md
# MAGIC List of run infos are obtained

# COMMAND ----------

mlflow.list_run_infos(experiment.experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC Last active run can be found out

# COMMAND ----------

mlflow.last_active_run()

# COMMAND ----------

# MAGIC %md
# MAGIC We can also search for runs using 'search_runs' and also obtain the data in sorted form in dataframe wrt some metric(here it is Test_accuracy)

# COMMAND ----------

df_run_metrics = mlflow.search_runs([experiment.experiment_id], order_by = ['metrics.Test_accuracy DESC'])

df_run_metrics

# COMMAND ----------

best_runs_df = df_run_metrics[df_run_metrics['metrics.AUC_score'] > 0.7]

best_runs_df

# COMMAND ----------

display(best_runs_df[['end_time', 'metrics.Test_f1_score']])
