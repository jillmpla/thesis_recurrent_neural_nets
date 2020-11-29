#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import Flare_Data

def generate_all_feature_years():

	################creates/loads csv with 2011's feature data##########
	current_di = os.getcwd()
	csv_for_2011 = "\\create_2011_features.csv"
	where_2011 = current_di + csv_for_2011

	if not os.path.exists(where_2011):
		import to_create_2011
		to_create_2011.get_2011_Features()
		df_2011 = pd.read_csv('create_2011_features.csv', index_col=0)
		df_2011_done = Flare_Data.convert_time_2011(df_2011)
		Flare_Data.save_some_features(df_2011_done) 

	if os.path.exists(where_2011):
		df_2011 = pd.read_csv('create_2011_features.csv', index_col=0)
		df_2011_done = Flare_Data.convert_time_2011(df_2011)
		Flare_Data.save_some_features(df_2011_done)
	####################################################################

	################creates/loads csv with 2012's feature data##########
	current_di = os.getcwd()
	csv_for_2012 = "\\create_2012_features.csv"
	where_2012 = current_di + csv_for_2012

	if not os.path.exists(where_2012):
		import to_create_2012
		to_create_2012.get_2012_Features()
		df_2012 = pd.read_csv('create_2012_features.csv', index_col=0)
		df_2012_done = Flare_Data.convert_time_2012(df_2012)
		Flare_Data.save_some_features(df_2012_done) 

	if os.path.exists(where_2012):
		df_2012 = pd.read_csv('create_2012_features.csv', index_col=0)
		df_2012_done = Flare_Data.convert_time_2012(df_2012)
		Flare_Data.save_some_features(df_2012_done)
	####################################################################

	################creates/loads csv with 2013's feature data##########
	current_di = os.getcwd()
	csv_for_2013 = "\\create_2013_features.csv"
	where_2013 = current_di + csv_for_2013

	if not os.path.exists(where_2013):
		import to_create_2013
		to_create_2013.get_2013_Features()
		df_2013 = pd.read_csv('create_2013_features.csv', index_col=0)
		df_2013_done = Flare_Data.convert_time_2013(df_2013)
		Flare_Data.save_some_features(df_2013_done) 

	if os.path.exists(where_2013):
		df_2013 = pd.read_csv('create_2013_features.csv', index_col=0)
		df_2013_done = Flare_Data.convert_time_2013(df_2013)
		Flare_Data.save_some_features(df_2013_done)
	####################################################################

	################creates/loads csv with 2014's feature data##########
	current_di = os.getcwd()
	csv_for_2014 = "\\create_2014_features.csv"
	where_2014 = current_di + csv_for_2014

	if not os.path.exists(where_2014):
		import to_create_2014
		to_create_2014.get_2014_Features()
		df_2014 = pd.read_csv('create_2014_features.csv', index_col=0)
		df_2014_done = Flare_Data.convert_time_2014(df_2014)
		Flare_Data.save_some_features(df_2014_done) 

	if os.path.exists(where_2014):
		df_2014 = pd.read_csv('create_2014_features.csv', index_col=0)
		df_2014_done = Flare_Data.convert_time_2014(df_2014)
		Flare_Data.save_some_features(df_2014_done)
	####################################################################

	################creates/loads csv with 2015's feature data##########
	current_di = os.getcwd()
	csv_for_2015 = "\\create_2015_features.csv"
	where_2015 = current_di + csv_for_2015

	if not os.path.exists(where_2015):
		import to_create_2015
		to_create_2015.get_2015_Features()
		df_2015 = pd.read_csv('create_2015_features.csv', index_col=0)
		df_2015_done = Flare_Data.convert_time_2015(df_2015)
		Flare_Data.save_some_features(df_2015_done) 

	if os.path.exists(where_2015):
		df_2015 = pd.read_csv('create_2015_features.csv', index_col=0)
		df_2015_done = Flare_Data.convert_time_2015(df_2015)
		Flare_Data.save_some_features(df_2015_done)
	####################################################################
	
	################creates/loads csv with 2016's feature data##########
	current_di = os.getcwd()
	csv_for_2016 = "\\create_2016_features.csv"
	where_2016 = current_di + csv_for_2016

	if not os.path.exists(where_2016):
		import to_create_2016
		to_create_2016.get_2016_Features()
		df_2016 = pd.read_csv('create_2016_features.csv', index_col=0)
		df_2016_done = Flare_Data.convert_time_2016(df_2016)
		Flare_Data.save_some_features(df_2016_done) 

	if os.path.exists(where_2016):
		df_2016 = pd.read_csv('create_2016_features.csv', index_col=0)
		df_2016_done = Flare_Data.convert_time_2016(df_2016)
		Flare_Data.save_some_features(df_2016_done)
	####################################################################
	
	return (print("CSVs compiled."))