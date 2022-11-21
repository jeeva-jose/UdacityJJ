# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
-	The provided data set contains marketing data about individuals. The classification goal is to predict whether the customer will subscribe a bank term deposit 
-	First the data set pass to  Scikit-learn based Logistic Regression hyperdrive then created tuned model 
-	The second step the same dataset passes to Azure AutoML to get the best model 


## Scikit-learn Pipeline
-	Created the workspace, cluster to train and predict the model.
-	Configured the RandomParameterSampling as a hyperparameter  
-	Defines an early termination policy based on slack criteria, and a frequency
-	Setup the sklearn environment for training  
-	We need to make it ready the training script making the following changes like, Created TabularDataset using TabularDatasetFactory , loaded dataset pass to clen up data and given to train and test data, The distribution of data is 80% data pass to train and the remaining 20% send to testing. LogisticRegression then create the final model based on the parameter
-	Hyperdrive take a parameter as ScriptRunConfig contain location of training script, target compute and environment details.
-	Once all the parameters are ready then start configuring the HyperDriveConfig total runs 20 and concurrent runs 4
-	Submit the hyperdrive config and wait for the result will predict the best model based on the configuration.
-	When the Hyperdrive result is ready to find the best model by calling get_best_run_by_primary_metric then save the model for further experiment.


## AutoML
-	Automated machine learning which can automate the pipeline processing and improve the efficiency of computing and save the effort. 
-	During training, Azure Machine Learning creates a number of pipelines in parallel that try different algorithms and predict the correct model. If it finds the correct model stop automatically.
-	Here I have used: classification


## Pipeline comparison
-	When I compared two models AutoML accuracy was 0.91812 better than Hyperdrive accuracy was 0.9088012
-	Auto ml architecture interact with model automatically and find best model but hyperdrive need to create the pipeline separately.
-	AutoML is the right architecture to deal with complex model predication.

## Future work
Looking for low code no code platform and plus customization. Make sure the data preparation, data quality, data validation and accuracy for the prediction. Auto suggest the compute power*

