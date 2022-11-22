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

### Benefits of parameter sampler

-	I have used RandomParameterSampling, that make a random sampling on hyperparameter search space.
-	In this case, I have defined choice function to generate a discrete set of values and uniform function to generate a distribution of continuous values. So, there is a more chance of increasing the accuracy of the model.
-	C and max_iter are the hyperparameters provided in the problem statement so that the hyperdrive can try all the options for the values of these hyperparameters can increase possible accuracy.

### Benefits of early stopping policy

-	Early Stopping policy help to terminate to run the hyperdrive run if the model is not improving the better accuracy by given amount of time and iteration. 
-	The main befits are reduce the time and saves a lot of computational resources
-	slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.
-	evaluation_interval: Optional. The frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.
-	delay_evaluation: Optional. The number of intervals to delay policy evaluation. Use this parameter to avoid premature termination of training runs. If specified, the policy applies every multiple of evaluation_interval that is greater than or equal to delay_evaluation.



## AutoML
-	Automated machine learning which can automate the pipeline processing and improve the efficiency of computing and save the effort. 
-	During training, Azure Machine Learning creates a number of pipelines in parallel that try different algorithms and predict the correct model. If it finds the correct model stop automatically.
-	Here I have used: classification

      ### The following parameters required to run AutoML
  
     - compute_target: A compute target is a designated compute resource or environment where you run your training script or host your service            deployment.
     - task : what task needs to be performed , regression or classification
     - training_data : the data on which we need to train the autoML.
     - label_column_name : the column name in the training data which is the output label.
     - iterations : the number of iterations we want to run AutoML.
     - primary_metric : the evaluation metric for the models
     - n_cross_validations : n-fold cross validations needed to perform in each model
     - experiment_timeout_minutes : the time in minutes after which autoML will stop.


## Pipeline comparison
-	When I compared two models AutoML accuracy was 0.91812 better than Hyperdrive accuracy was 0.9088012
-	Auto ml architecture interact with model automatically and find best model but hyperdrive need to create the pipeline separately.
-	AutoML is the right architecture to deal with complex model predication.

## Future work
Looking for low code no code platform and plus customization. Make sure the data preparation, data quality, data validation and accuracy for the prediction. Auto suggest the compute power*

