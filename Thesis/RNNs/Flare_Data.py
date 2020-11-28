#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
import os
import glob
import shutil
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from astropy.time import Time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import drms #https://pypi.org/project/drms/

all_of_the_Dataframe = pd.DataFrame()

def get_some_features():
	return(all_of_the_Dataframe)
	
def save_some_features(to_save):
	global all_of_the_Dataframe
	all_of_the_Dataframe = all_of_the_Dataframe.append(to_save)
	return()
	
def convert_time_2012(k):
	#change T_REC to datetime type
	k.T_REC = drms.to_datetime(k.T_REC)
	
	#convert tai time to utc
	t1 = Time(k.T_REC, format='datetime64', scale='tai')
	t2 = t1.utc
	t3 = t2.iso
	k.T_REC = t3
	k.T_REC = pd.to_datetime(k.T_REC)
	
	#delete first row of df from previous year
	k = k[(k['T_REC'].dt.year != 2011)]
	return(k)

def convert_time_2013(k):
	#change T_REC to datetime type
	k.T_REC = drms.to_datetime(k.T_REC)
	
	#convert tai time to utc
	t1 = Time(k.T_REC, format='datetime64', scale='tai')
	t2 = t1.utc
	t3 = t2.iso
	k.T_REC = t3
	k.T_REC = pd.to_datetime(k.T_REC)
	
	#delete first row of df from previous year
	k = k[(k['T_REC'].dt.year != 2012)]
	return(k)

def convert_time_2014(k):
	#change T_REC to datetime type
	k.T_REC = drms.to_datetime(k.T_REC)
	
	#convert tai time to utc
	t1 = Time(k.T_REC, format='datetime64', scale='tai')
	t2 = t1.utc
	t3 = t2.iso
	k.T_REC = t3
	k.T_REC = pd.to_datetime(k.T_REC)
	
	#delete first row of df from previous year
	k = k[(k['T_REC'].dt.year != 2013)]
	return(k)
	
def convert_time_2015(k):
	#change T_REC to datetime type
	k.T_REC = drms.to_datetime(k.T_REC)
	
	#convert tai time to utc
	t1 = Time(k.T_REC, format='datetime64', scale='tai')
	t2 = t1.utc
	t3 = t2.iso
	k.T_REC = t3
	k.T_REC = pd.to_datetime(k.T_REC)
	
	#delete first row of df from previous year
	k = k[(k['T_REC'].dt.year != 2014)]
	return(k)

def getAllData(binary):
	#input - hmi.sharp_720s from JSOC:::
	#http://jsoc.stanford.edu/doc/data/hmi/sharp/sharp.htm
	#-------------------------------------------------------------------------#
	#-------------------------------------------------------------------------#
	#-------------------------------------------------------------------------#
	k = get_some_features()
	
	k = k[k['NOAA_AR'] != 0]
	k = k.dropna()
	k = k.drop_duplicates()
	
	k.rename(columns={'T_REC': 'DATE_TIME'}, inplace=True)
	k.sort_values('DATE_TIME', inplace=True, ascending=True)
	k = k.reset_index(drop=True)
	
	print('The time series starts from: ', k['DATE_TIME'].min())
	print('The time series ends on: ', k['DATE_TIME'].max())

	#label - GOES flare events:::
	#ftp://ftp.swpc.noaa.gov/pub/warehouse/
	#-------------------------------------------------------------------------#
	#-------------add/remove .txt files/days from Data folder-----------------#
	#-------------------------------------------------------------------------#
	#time in UTC
	#Event - arbitrary event number assigned by SWPC
	#Reg# - SWPC-assigned solar region number
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#!!!!!If any additional daily report years are added to the Data folder add a new entry here!!!!!
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	df = pd.DataFrame()
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	cwd = os.getcwd() 
	daily_Events = "\\Data"
	my_dir = cwd + daily_Events
	os.chdir(my_dir)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	daily_reports_year = "\\2012" #folder year for daily reports, only variable changed in this block
	year_dir = my_dir + daily_reports_year
	os.chdir(year_dir)

	filesList = []
	for files in glob.glob("*.txt"):
		filesList.append(files)
    
	for aFile in filesList:
		frame = pd.read_csv(aFile, skiprows=11, engine = 'python', sep='[\s+]{2,}', skip_blank_lines=True, header=None, names=["Event", "Begin", "Max", "End", "Obs", "Q", "Type", "Loc/Frq", "Particulars", "Particulars2", "Reg#"], usecols = [0,1,2,3,4,5,6,7,8,9,10])
		frame['Filename'] = Path(aFile).stem[:-6]
		frame['Day'] = pd.to_datetime(frame['Filename'])    
		df = df.append(frame)
	os.chdir('..')
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	daily_reports_year = "\\2013" #folder year for daily reports, only variable changed in this block
	current_dir = os.getcwd() 
	year_dir = current_dir + daily_reports_year
	os.chdir(year_dir)

	filesList = []
	for files in glob.glob("*.txt"):
		filesList.append(files)

	for aFile in filesList:
		frame = pd.read_csv(aFile, skiprows=11, engine = 'python', sep='[\s+]{2,}', skip_blank_lines=True, header=None, names=["Event", "Begin", "Max", "End", "Obs", "Q", "Type", "Loc/Frq", "Particulars", "Particulars2", "Reg#"], usecols = [0,1,2,3,4,5,6,7,8,9,10])
		frame['Filename'] = Path(aFile).stem[:-6]
		frame['Day'] = pd.to_datetime(frame['Filename'])    
		df = df.append(frame)
	os.chdir('..')
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	daily_reports_year = "\\2014" #folder year for daily reports, only variable changed in this block
	current_dir = os.getcwd() 
	year_dir = current_dir + daily_reports_year
	os.chdir(year_dir)

	filesList = []
	for files in glob.glob("*.txt"):
		filesList.append(files)

	for aFile in filesList:
		frame = pd.read_csv(aFile, skiprows=11, engine = 'python', sep='[\s+]{2,}', skip_blank_lines=True, header=None, names=["Event", "Begin", "Max", "End", "Obs", "Q", "Type", "Loc/Frq", "Particulars", "Particulars2", "Reg#"], usecols = [0,1,2,3,4,5,6,7,8,9,10])
		frame['Filename'] = Path(aFile).stem[:-6]
		frame['Day'] = pd.to_datetime(frame['Filename'])    
		df = df.append(frame)
	os.chdir('..')
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	daily_reports_year = "\\2015" #folder year for daily reports, only variable changed in this block
	current_dir = os.getcwd() 
	year_dir = current_dir + daily_reports_year
	os.chdir(year_dir)

	filesList = []
	for files in glob.glob("*.txt"):
		filesList.append(files)

	for aFile in filesList:
		frame = pd.read_csv(aFile, skiprows=11, engine = 'python', sep='[\s+]{2,}', skip_blank_lines=True, header=None, names=["Event", "Begin", "Max", "End", "Obs", "Q", "Type", "Loc/Frq", "Particulars", "Particulars2", "Reg#"], usecols = [0,1,2,3,4,5,6,7,8,9,10])
		frame['Filename'] = Path(aFile).stem[:-6]
		frame['Day'] = pd.to_datetime(frame['Filename'])    
		df = df.append(frame)
	os.chdir('..')
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	#get Peak_Time, Class, and Region_Num from data
	df = df[~df.Event.str.contains('#')]
	df = df[df['Type'] == "XRA"]
	df['Reg#'] = df['Reg#'].fillna(0)
	del df['Obs']
	del df['Q']
	del df['Type']
	del df['Loc/Frq']
	del df['Particulars2']
	del df['Filename']
	del df['Begin']
	del df['End']
	df.rename(columns={'Particulars': 'Class'}, inplace=True)
	df['Class'] = df['Class'].str.replace('\d+', '')
	df['Event'] = df['Event'].astype('int64')
	df['Reg#'] = df['Reg#'].astype('int64')
	df['Class'] = df['Class'].astype('str')
	df['Class'] = df['Class'].str.replace(r'.', '')
	df['Max'] = pd.to_numeric(df['Max'], errors='coerce').fillna(0)
	df['Max'] = pd.to_timedelta(df['Max'] // 100, unit='H') + pd.to_timedelta(df['Max'] % 100, unit='T')
	df['Peak_Time'] = df['Day'] + df['Max']
	del df['Day']
	del df['Max']
	del df['Event']
	df.rename(columns={'Reg#': 'Region_Num'}, inplace=True)
	df = df[df['Region_Num'] != 0]
	colTitles = ['Peak_Time', 'Class', 'Region_Num']
	df = df.reindex(columns=colTitles)

	#Add ~Connector~ GOES XRS Report:::
	#https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/
	#-------------------------------------------------------------------------#
	#-----------add/remove .txt year(s) from GOES_XRS_Reports folder----------#
	#-------------------------------------------------------------------------#
	#time in UTC
	cwd = os.getcwd() 
	extraEventsXRS = "\\GOES_XRS_Reports"
	my_dir = cwd + extraEventsXRS
	filesListXRS = []
	dataframesXRS = []
	os.chdir( my_dir )

	for filesXRS in glob.glob("*.txt"):
		filesListXRS.append(filesXRS)

	dfXRS = pd.DataFrame()

	for aFile in filesListXRS:
		frame = pd.read_csv(aFile, engine = 'python', sep='[\s+]{2,}', skip_blank_lines=True, header=None, names= ["Zero","One","Two","Three","Four","Five"],usecols=[0,1,2,3,4,5])
		dfXRS = dfXRS.append(frame)

	#get Peak_Time and NOAA_AR from data
	del dfXRS["Two"]
	del dfXRS["Three"]
	del dfXRS["Five"]
	dfXRS['Zero'] = dfXRS['Zero'].astype('str')
	dfXRS['One'] = dfXRS['One'].astype('str')
	dfXRS['Four'] = dfXRS['Four'].astype('str')
	dfXRS['One'] = dfXRS['One'].str.split().str[2]
	dfXRS['Four'] = dfXRS['Four'].str.split().str[1]
	dfXRS['Zero'] = dfXRS['Zero'].map(lambda x: str(x)[-6: ])
	dfXRS['Zero'] = '20' + dfXRS['Zero'].astype(str)
	dfXRS['Zero'] = pd.to_datetime(dfXRS['Zero'])
	dfXRS['One'] = pd.to_numeric(dfXRS['One'], errors='coerce').fillna(0)
	dfXRS['One'] = pd.to_timedelta(dfXRS['One'] // 100, unit='H') + pd.to_timedelta(dfXRS['One'] % 100, unit='T')
	dfXRS['Peak_Time'] = dfXRS['Zero'] + dfXRS['One']
	dfXRS["Four"] = dfXRS["Four"].fillna(0)
	dfXRS.rename(columns={'Four': 'NOAA_AR'}, inplace=True)
	del dfXRS["Zero"]
	del dfXRS["One"]
	dfXRS = dfXRS[dfXRS['NOAA_AR'] != 0]
	colTitles = ['Peak_Time', 'NOAA_AR']
	dfXRS = dfXRS.reindex(columns=colTitles)

	#merge df and dfXRS on Peak_Time
	Labs = df.merge(dfXRS, on='Peak_Time')

	#copy of dataframe Labs
	Labs1 = Labs
	Labs1['Class'] = Labs1['Class'].astype('str')
	Labs1['Date'] = pd.to_datetime(Labs1['Peak_Time']).dt.date
	del Labs1["Peak_Time"]

	#5 flare classes, A is smallest, X is largest
	key = {'A': 5, 'B': 4, 'C': 3, 'M': 2, 'X': 1}
	Labs1['Class_Val'] = Labs1['Class'].map(key)

	#group by Date, filter by Class_Val min
	Labs1 = Labs1.loc[Labs1.groupby('Date')['Class_Val'].idxmin()]

	#add day before dates
	Labs1['Date_Day_Before'] = pd.to_datetime(Labs1['Date']).apply(pd.DateOffset(-1))

	#make sure column(s) are converted
	Labs1.Date = pd.to_datetime(Labs1.Date)
	Labs1.Date_Day_Before = pd.to_datetime(Labs1.Date_Day_Before)

	#organize columns
	colTitlesNew = ['Date', 'Date_Day_Before', 'Class_Val', 'Region_Num', 'NOAA_AR']
	Labs1 = Labs1.reindex(columns=colTitlesNew)

	#remap 5 flare classes, A is smallest, X is largest
	key = {5 :'A', 4 :'B', 3 :'C', 2 :'M', 1 :'X'}
	Labs1['Class_Val'] = Labs1['Class_Val'].map(key)

	#make copy of Labs1
	new_labs = Labs1

	#make sure Class_Val are strings
	new_labs['Class_Val'] = new_labs['Class_Val'].astype(str)

	#make copy of k to work with
	k1 = k
	k1['Date'] = pd.to_datetime(k['DATE_TIME']).dt.date

	#organize columns, reset index
	col_Titles_New = ['DATE_TIME', 'Date', 'HARPNUM', 'NOAA_AR', 'TOTUSJH', 'TOTUSJZ', 'SAVNCPP', 'USFLUX', 'ABSNJZH', 'TOTPOT', 'SIZE_ACR', 'NACR', 'MEANPOT', 'SIZE', 'MEANJZH', 'SHRGT45', 'MEANSHR', 'MEANJZD', 'MEANALP', 'MEANGBT', 'MEANGAM', 'MEANGBZ', 'MEANGBH', 'NPIX']
	k1 = k1.reindex(columns=col_Titles_New)
	k1.reset_index(drop=True, inplace=True)
	
	new_k_data = k1
	
	#drop duplicate rows
	new_k_data = new_k_data.drop_duplicates()

	#reset index
	new_k_data.reset_index(drop=True, inplace=True)

	#remove date+AR with < 90 records (75% of 120 [every 12 minutes] max entries per day, per AR) 
	#to ensure a good random sample per day, per AR
	count_Ds = new_k_data.groupby(['Date', 'NOAA_AR']).filter(lambda x : len(x)>113)
	count_Ds.reset_index(drop=True, inplace=True)

	#-------------------------------------------------------------------------#
	#input (new_k_data_f) - sharp summary parameters
	#labels (new_labs) - maximum flare class produced by the AR 
	#in the next 24 hours after the end time of a sequence
	#-------------------------------------------------------------------------#

	#get :::90::: random rows per day, per AR
	np.random.seed(0)
	count_Ds = count_Ds.astype({"Date": str})
	COLS = ['DATE_TIME', 'HARPNUM', 'TOTUSJH', 'TOTUSJZ', 'SAVNCPP', 'USFLUX', 'ABSNJZH', 'TOTPOT', 'SIZE_ACR', 'NACR', 'MEANPOT', 'SIZE', 'MEANJZH', 'SHRGT45', 'MEANSHR', 'MEANJZD', 'MEANALP', 'MEANGBT', 'MEANGAM', 'MEANGBZ', 'MEANGBH', 'NPIX']
	new_k_data_f = count_Ds.groupby(['Date', 'NOAA_AR'])[COLS].apply(pd.Series.sample, n=114, replace=False).reset_index()

	#order by DATE_TIME
	new_k_data_f.sort_values('DATE_TIME')

	#delete level_2 column
	del new_k_data_f["level_2"]

	allData = []
	allLabels = []
	a_control = 0
	a_control_2 = 0
	a_control_3 = 0

	grouped_data = new_k_data_f.groupby(['Date', 'NOAA_AR'])
	
	#loop that creates groups of feature data by day, if flare occurrence next day
	for g in grouped_data.groups:
		mygroup = grouped_data.get_group(g)
		mydata = mygroup[['Date', 'NOAA_AR']][:1].to_numpy()
		mylabel = new_labs.loc[(new_labs['Date_Day_Before'] == mydata[0,0])] 
		mylabel_1 = mylabel.loc[(mylabel['NOAA_AR'] == str(mydata[0,1]))]
        #create allLabels list~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if not mylabel_1.empty:
			fla_class = mylabel_1['Class_Val']
			fla_class_1 = ' '.join(map(str, fla_class))
			if binary == True:
				if fla_class_1 == 'A':
					continue
				if fla_class_1 == 'B':
					continue
				if fla_class_1 == 'X':
					continue
				else:
					allLabels.append('F')
			if binary == False:
				if fla_class_1 == 'A':
					continue
				if fla_class_1 == 'B':
					continue
				if fla_class_1 == 'X':
					continue
				if fla_class_1 == 'C':
					if a_control % 2 == 0:
						allLabels.append('C')
						a_control += 1
					else: 
						a_control += 1
						continue
				else:
					allLabels.append(fla_class_1)
		if mylabel_1.empty:
			if binary == True:
				if a_control_2 % 10 == 0:
					no_fla = 'N'
					allLabels.append(no_fla)
					a_control_2 += 1
				else:
					a_control_2 += 1
					continue
			if binary == False:
				if a_control_3 % 22 == 0:
					no_fla = 'N'
					allLabels.append(no_fla)
					a_control_3 += 1
				else:
					a_control_3 += 1
					continue
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		del mygroup['Date']
		del mygroup['NOAA_AR']
		del mygroup['HARPNUM']
		del mygroup['DATE_TIME']
        #normalize features per group/day+AR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		min_max_scale = MinMaxScaler()
		scaled = min_max_scale.fit_transform(mygroup.astype('float64'))
		normalized_mygroup = pd.DataFrame(scaled)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		allData.append(normalized_mygroup.to_numpy())
        #check inverse~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #invers = min_max_scale.inverse_transform(normalized)
        #n_inversed = pd.DataFrame(invers)
        #n_inversed.describe().apply(lambda p: p.apply(lambda k: format(k, 'g')))
        #print(n_inversed)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	allData = np.array(allData)
	allLabels = np.array(allLabels)

	num_sequences = allData.shape[0] #number of sequences
	tim_steps = allData.shape[1] #number of timesteps
	n_feats = allData.shape[2] #number of features per timestep

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#one-hot encoding----------------------------------------------------------------------------------
	#6 flare classes: N is no flare, A is smallest, X is largest, N is no flare
	#key = {'A', 'B', 'C', 'M', 'N', 'X'}
	a_encoder = LabelBinarizer()
	allLabels_enc = a_encoder.fit_transform(allLabels)
	#number of flare classes actually found in overall dataset
	count_of_classes = len(a_encoder.classes_)
	#test to see flare classes
	print(np.unique(allLabels, return_counts=True))
	#check inverse
	#print(a_encoder.inverse_transform(allLabels_enc))
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	#normalized feature data - :::allData::: [num_sequences, number of timesteps, number of features per timestep]
	#encoded labels - :::allLabels_enc::: one-hot encoding

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#split into train and test - 50%(30%) train, 50% validation (of train), 20% test------------------------
	#use specific seed for pseudo-random number generator when spliting data to properly compare machine learning models
	#get a balanced number of examples for each class label in both train and test with stratify=y
	X_train, X_test, y_train, y_test = train_test_split(allData, allLabels_enc, test_size = 0.20, random_state=1, stratify=allLabels_enc)
	#split train set further into train and validation sets
	X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.50, random_state=1, stratify=y_train)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	return(X_train, X_val, X_test, y_train, y_val, y_test, tim_steps, n_feats, count_of_classes, a_encoder)
