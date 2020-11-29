#!/usr/bin/env python
# coding: utf-8

#only run once~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#creates csv with :::2010's::: feature data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#:::hmi.sharp_720s from JSOC:::
#http://jsoc.stanford.edu/doc/data/hmi/sharp/sharp.htm

import pandas as pd
import drms #https://pypi.org/project/drms/

#compile :::2010::: feature dataframe
#multiple queries needed due to record return limit

def get_2010_Features():

	h = drms.Client()

	k = h.query('hmi.sharp_720s[][2010.01.01_TAI/7d]', key='T_REC, HARPNUM, NOAA_AR, TOTUSJH, TOTUSJZ, SAVNCPP, USFLUX, ABSNJZH, TOTPOT, SIZE_ACR, NACR, MEANPOT, SIZE, MEANJZH, SHRGT45, MEANSHR, MEANJZD, MEANALP, MEANGBT, MEANGAM, MEANGBZ, MEANGBH, NPIX')
	f_dataframe = k

	k = h.query('hmi.sharp_720s[][2010.01.08_TAI/7d]', key='T_REC, HARPNUM, NOAA_AR, TOTUSJH, TOTUSJZ, SAVNCPP, USFLUX, ABSNJZH, TOTPOT, SIZE_ACR, NACR, MEANPOT, SIZE, MEANJZH, SHRGT45, MEANSHR, MEANJZD, MEANALP, MEANGBT, MEANGAM, MEANGBZ, MEANGBH, NPIX')
	f_dataframe = f_dataframe.append(k)
	
	k = h.query('hmi.sharp_720s[][2010.01.15_TAI/7d]', key='T_REC, HARPNUM, NOAA_AR, TOTUSJH, TOTUSJZ, SAVNCPP, USFLUX, ABSNJZH, TOTPOT, SIZE_ACR, NACR, MEANPOT, SIZE, MEANJZH, SHRGT45, MEANSHR, MEANJZD, MEANALP, MEANGBT, MEANGAM, MEANGBZ, MEANGBH, NPIX')
	f_dataframe = f_dataframe.append(k)
	










	f_dataframe.to_csv('create_2010_features.csv')
	
	return()

