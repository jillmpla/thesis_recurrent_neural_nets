#!/usr/bin/env python
# coding: utf-8

import os
import shutil

def make_clear_tensorboard_folder():
	#removes old tensorboard Logs and recreates or creates Logs folder
	current_dir = os.getcwd()
	l_folder = "\\Logs"
	where_logs = current_dir + l_folder

	if not os.path.exists(where_logs):
		os.mkdir(where_logs)
    
	if os.path.exists(where_logs):
		shutil.rmtree(where_logs)
		os.mkdir(where_logs)
	
	return(where_logs)