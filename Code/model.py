print("Loading in required modules...", end=' ')
import platform
from os import listdir
import pandas
import numpy
from darts import TimeSeries
from darts.models import NBEATSModel, RandomForest
from darts.metrics import rmse
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
import torchvision

# from pytorch_lightning.callbacks import EarlyStopping
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.integration.pytorch_lightning import TuneReportCallback
# from ray.tune.schedulers import ASHAScheduler
# from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError, MetricCollection



DATA_PATH = "Data/Tensor/"
CRIMES_MEAN = 12.50395
CRIMES_SD = 19.18644



# reverse standardisation process for final crime data for more intelligible forecasts in real terms
def destandardise(a):
	return (a * CRIMES_SD) + CRIMES_MEAN

# turn a sequence of 1x1x1 time series into an array, optionally destandardising its data
def flattenTSSeq(a, destand = False):
	if destand:
		return list(map(lambda x: destandardise(x.first_value()), a ))
	else:
		return list(map(lambda x: x.first_value(), a))

# calculate RMSE and MAE values for two flat arrays
def getMetrics(actual, predicted):
	return [
		numpy.sqrt(  	numpy.mean(numpy.square(	numpy.subtract(predicted, actual) )) ),	# RMSE
				numpy.mean(numpy.absolute(	numpy.subtract(predicted, actual) )),	# MAE
	]



def main():

	# read in from CSVs and store in dictionary by LSOA with formatted columns
	print("Done.\nLoading in dataset...", end=' ')
	lsoas = []
	crimes = {}
	for f in listdir(DATA_PATH):
		if f[0] != '.':
			lsoas.append(f[:-4])
			# cast float64 -> float32 for mts acceleration compatibility
			curSeries = TimeSeries.from_csv( (DATA_PATH+f) ).with_columns_renamed("Unnamed: 0", "months").astype("float32")
			crimes[f[:-4]+"-target"] = curSeries["crimes"]
			crimes[f[:-4]+"-covariates"] = curSeries["housePrices"].stack(curSeries["population"]).stack(curSeries["femProportion"]).stack(curSeries["ythProportion"]).stack(curSeries["pubs"])
	
	# split LSOAs with 80/20 train/test proportions respectively
	print("Done.\nSplitting data by train/test and target/covariate...", end=' ')
	splitTrainTest = train_test_split( lsoas, train_size=0.8 )
	
	# map to ordered sequences for later fitting
	trainTargets = list(map(lambda l: crimes[l+"-target"], splitTrainTest[0]))
	trainCovariates = list(map(lambda l: crimes[l+"-covariates"], splitTrainTest[0]))
	testTargets = list(map(lambda l: crimes[l+"-target"], splitTrainTest[1]))
	testCovariates = list(map(lambda l: crimes[l+"-covariates"], splitTrainTest[1]))

	print("Done.\nDefining models...", end=' ')
	# define and fit N-BEATS model
	# def fitNBEATS():
		modelNBEATS = NBEATSModel(
			save_checkpoints	= True,		# for resumption in case of interruption
			input_chunk_length	= 60,		# 5-year lookback time
			output_chunk_length	= 1,		# only predict one month into the future
			generic_architecture	= False,	# use interpretable model
			num_blocks		= 3,		# use 3 blocks per trend/seasonality stack
			pl_trainer_kwargs	= {		# allow multi-gpu acceleration in training
				"accelerator":		"gpu",
				"devices":		-1
			}
		)
		modelNBEATS.fit(
			series			= trainTargets,
			past_covariates		= trainCovariates,
			val_series		= testTargets,
			val_past_covariates	= testCovariates,
			verbose			= True
		)
	
	# define N-BEATS model without covariates
	modelNBEATSNoCovariates = NBEATSModel(
		save_checkpoints	= True,		# for resumption in case of interruption
		input_chunk_length	= 60,		# 5-year lookback time
		output_chunk_length	= 1,		# only predict one month into the future
		generic_architecture	= False,	# use interpretable model
		num_blocks		= 3,		# use 3 blocks per trend/seasonality stack
		pl_trainer_kwargs	= {		# allow multi-gpu acceleration in training
			"accelerator":		"gpu",
			"devices":		-1
		}
	)

	# define random forest model
	modelRandomForest = RandomForest(
		output_chunk_length	= 1,		# only predict one month into the future
		lags			= 1,		# use static lags
		lags_past_covariates	= 1
	)

	# define random forest model without covariates
	modelRandomForestNoCovariates = RandomForest(
		output_chunk_length	= 1,		# only predict one month into the future
		lags			= 1		# use static lags
	)	

	# fit...
	print("Done.\nTraining models...", end=' ')
	
	modelNBEATSNoCovariates.fit(
		series			= trainTargets,
		val_series		= testTargets,
		verbose			= True
	)
	modelNBEATSNoCovariates.save("Code/nbeats_no_covariates.pt")

	modelRandomForest.fit(
		series			= trainTargets,
		past_covariates		= trainCovariates
	)
	modelRandomForest.save("Code/randomforest.pt")

	modelRandomForestNoCovariates.fit(
		series			= trainTargets
	)
	modelRandomForestNoCovariates.save("Code/randomforest_no_covariates.pt")

	# print("Hyperparameter tuning...", end='')
	# analysis = tune.run(
	# 	tune.with_parameters(
	# 		fitNBEATS,
	# 		callbacks = [
	# 			EarlyStopping(
	# 				monitor = "val_MeanAbsolutePercentageError",
	# 				patience = 5,
	# 				min_delta = 0.05,
	# 				mode = "min"
	# 			),
	# 			TuneReportCallback(
	# 				{
	# 					"loss": "val_Loss",
	# 					"MAPE": "val_MeanAbsolutePercentageError",
	# 				},
	# 				on = "validation_end"
	# 			)
	# 		],
	# 		train = trainTargets,
	# 		val = testTargets
	# 	),
	# 	resources_per_trial = {
	# 		"cpu": 6,
	# 		"gpu": 9
	# 	},
	# 	metric = "MAPE",
	# 	mode = "min",
	# 	config = {
	# 		"batch_size": tune.choice([16, 32, 64, 128]),
	# 		"num_blocks": tune.choice([1, 2, 3, 4, 5]),
	# 		"num_stacks": tune.choice([32, 64, 128]),
	# 		"dropout": tune.uniform(0, 0.2)
	# 	},
	# 	num_samples = 10,
	# 	scheduler = ASHAScheduler(
	# 		max_t = 1000,
	# 		grace_period = 3,
	# 		reduction_factor = 2
	# 	),
	# 	progress_reporter = CLIReporter(
	#		parameter_columns = [
	# 			"batch_size",
	# 			"num_blocks",
	# 			"num_stacks",
	# 			"dropout"
	# 		],
	# 		metric_columns = [
	# 			"loss",
	# 			"MAPE",
	# 			"training_iteration"
	# 		]
	# 	),
	# 	name = "tuner"
	# )
	# print(analysis)

	# split final month from test set for predictions
	print("Done.\nPreparing test set for prediction...", end=' ')
	testTargetsPast = []
	testTargetsFuture = []
	testCovariatesPast = testCovariates.copy()
	for lsoa in range(len(testTargets)):
		curSplit = testTargets[lsoa].split_before(testTargets[lsoa].duration)
		testTargetsPast.append(curSplit[0])
		testTargetsFuture.append(curSplit[1])
		testCovariatesPast[lsoa] = testCovariates[lsoa].drop_after(testCovariates[lsoa].duration)		

	# predict
	print("Done.\nMaking predictions...", end=' ')
	forecasts = {}
	forecasts["N-BEATS"] = modelNBEATS.load("Code/nbeats.pt").predict(
		series		= testTargetsPast,
		past_covariates	= testCovariatesPast,
		n		= 1,			# just predict one month into the future
		n_jobs		= -1,			# enable parallelisation
	)

	forecasts["N-BEATS (no covariates)"] = modelNBEATSNoCovariates.load("Code/nbeats_no_covariates.pt").predict(
		series		= testTargetsPast,
		n		= 1,			# just predict one month into the future
		n_jobs		= -1,			# enable parallelisation
	)

	forecasts["Random forest"] = modelRandomForest.load("Code/randomforest.pt").predict(
		series		= testTargetsPast,
		past_covariates	= testCovariatesPast,
		n		= 1			# just predict one month into the future
	)

	forecasts["Random forest (no covariates)"] = modelRandomForestNoCovariates.load("Code/randomforest_no_covariates.pt").predict(
		series		= testTargetsPast,
		n		= 1			# just predict one month into the future
	)

	# compile metrics and actual data into output
	print("Done.\nCompiling results...", end=' ')
	testTargetsFuture = flattenTSSeq(testTargetsFuture, destand=True)
	for model in forecasts.keys():
		forecasts[model] = flattenTSSeq(forecasts[model], destand=True)

	predictionDF = pandas.DataFrame({
		"LSOA": 					splitTrainTest[1],
		"Actual": 					testTargetsFuture,
		"Predicted (random forest, crime-intrinsic)": 	forecasts["Random forest (no covariates)"],
		"Predicted (random forest, with covariates)": 	forecasts["Random forest"],
		"Predicted (N-BEATS, crime-intrinsic)": 	forecasts["N-BEATS (no covariates)"],
		"Predicted (N-BEATS, with covariates)": 	forecasts["N-BEATS"]
	})
	metricsDF = pandas.DataFrame({
		"LSOA":						["RMSE", "MAE"],
		"Actual":					[0] * 2,
		"Predicted (random forest, crime-intrinsic)":	getMetrics(testTargetsFuture, forecasts["Random forest (no covariates)"]),
		"Predicted (random forest, with covariates)":	getMetrics(testTargetsFuture, forecasts["Random forest"]),
		"Predicted (N-BEATS, crime-intrinsic)":		getMetrics(testTargetsFuture, forecasts["N-BEATS (no covariates)"]),
		"Predicted (N-BEATS, with covariates)":		getMetrics(testTargetsFuture, forecasts["N-BEATS"])
	})
	predictionDF = pandas.concat([metricsDF, predictionDF])
	predictionDF.set_index("LSOA", inplace=True)
	predictionDF.to_csv("Data/predictions.csv")
	print("Done.")



# wrapper necessary for torch multiprocessing as import time processes will crash
if __name__ == "__main__":
	
	# check for correct processor utilisation
	print("Done.\nInitialising model preparation process...", end=' ')
	if not platform.processor() == "arm":
		print("\nRunning as if Intel processor. Please double check for ARM environment. Exiting.")
		exit()
	if not torch.backends.mps.is_available():
		print("\nLaptop GPU seems to have fallen out. Exiting.")
		exit()
	
	main()
