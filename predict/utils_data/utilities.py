#
# Stock prediction and prepare the traning and testing data
# 
# Author: JohnnyHaan
# Data: May_17Th_2020
#
# 

import pandas as pd 
import numpy as np
import os
import glob
import random
import math


keep_threshold = 0
keep_threshold_neg = -0.2

def read_useful_files(data_path, split = "train-sim"):
	# Paths
	path_ = os.path.join(data_path, split)
	print(path_)
	file_list = []
	files = glob.glob(path_ + '/*.csv')
	random.shuffle (files)
	for file in files:
		# apath = os.path.join(root, file)#合并成一个完整路径            
		if "sh.000" in str(file) or "sz.399" in str(file) :
			continue
		file_list.append(file)                                
	return file_list


def read_stock_data_items_and_transform(data_path, split = "train", n_dim_feautures = 12, hold_days = 5, predict_days = 5, gain_threshold = 2.0):
	""" read stock data then reform them"""
	# fixed params
	n_predict = 2   
	# filter = ['open','high','low','close','volume','amount','adjustflag','turn','tradestatus','pctChg','peTTM','pbMRQ','psTTM']    
	filter_data = ['open','high','low','close','volume','amount','turn','pctChg']   
	filter_info = ['open','high','low','close','volume','amount','turn','pctChg','code'] 
	# Read stock data    
	file_lists = read_useful_files(data_path, split)
	group = [] 
	labels = []
	tommorow_prices = []  
	today_prices = []    
	stock_code = []     
	each = np.zeros((1, hold_days, n_dim_feautures))
	Label = np.zeros(n_predict) 
	for csv in file_lists:
		dat_ = pd.read_csv(csv)
		
		## handle the exception
		for filt in filter_data:                       
			median = dat_[filt].median()
			dat_[filt].fillna(median, inplace = True)        
		datas = dat_[filter_info][:].values
		del_list = np.where(dat_['volume'] == 0)
		datas = np.delete(datas, del_list, axis=0) 
		## handle the exception end
		## 新亿 深深房A 金亚科技 .ST德奥
		rows = len(datas)
		if rows <= (hold_days + 1):
			continue
		## tranform the data  price = price / pre_close_price for the ones of open close high low
		#print(csv)
		#print(rows)        
		pre_close = datas[0:1, 3:4][0][0]        
		for i in range(1, rows) :
			## price record
			if (i < rows -hold_days - predict_days):
				Label_price = datas[i+hold_days:i+hold_days+1,1:2][0][0]
				tommorow_prices.append(Label_price)
				Label_price = datas[i+hold_days-1:i+hold_days,1:2][0][0]                
				today_prices.append(Label_price)
			## reform the price
			pre_close_hold = datas[i:i+1, 3:4][0][0]            
			datas[i:i+1, 0:4] = (datas[i:i+1, 0:4] - pre_close) / pre_close
			pre_close = pre_close_hold    
            
           
    
		## handle the labels  and the last predict_days of pc_changes ,if more than 2 (2%) as postiave 1 label others as 0   
		each_labels_num = rows - hold_days - predict_days
		for i in range(1, each_labels_num) :
			## train data  100000 for solveing forcas the error "OverflowError: Python int too large to convert to C long"
			mean_ = np.mean(datas[i:i+hold_days, 4:6]/100000, axis=0)
			std_ = np.std((datas[i:i+hold_days, 4:6]/100000).astype(np.int32), axis=0)
			if std_[0:1] == 0 or std_[1:2] == 0:
				continue
			datas[i:i+hold_days, 4:6] = (datas[i:i+hold_days, 4:6]/100000 - mean_) / std_              
			each = datas[i:i + hold_days,:]
			each[each==np.nan] = 0                        
			## normalize the volue and amount
			group.append(each[:,0:7])
			
			## labels           
			ratio_record = datas[i+hold_days:i+hold_days+predict_days, 7:8]
			#print(ratio_record)
			sum_all = np.sum(ratio_record, axis=0)
			#print(sum_all)            
			if sum_all > gain_threshold :
				Label = 1
			else :
				Label = 0 
			labels.append(Label)
			## price record
			## info record
			code = datas[i+hold_days:i+hold_days+1,8:9]  
			stock_code.append(code)
	#print(group)            
	return np.array(group), np.array(labels), np.array(tommorow_prices), np.array(today_prices), np.array(stock_code)



def read_stock_data_items_and_transform_by_stock_num(data_path, stock_code, n_dim_feautures = 12, hold_days = 5, predict_days = 5, gain_threshold = 2.0):
	""" Read data """
	# Fixed params
	n_predict = 2
	filter_data = ['open','high','low','close','volume','amount','turn','pctChg']   
	filter_info = ['open','high','low','close','volume','amount','turn','pctChg','code'] 
	# Read time-series data    
	file_lists = []
	file_lists.append(data_path + stock_code)
	group = [] 
	labels = []
	tommorow_prices = []  
	today_prices = []    
	stock_code = []     
	each = np.zeros((1, hold_days, n_dim_feautures))
	Label = np.zeros(n_predict) 
	for csv in file_lists:
		dat_ = pd.read_csv(csv)
		
		## handle the exception
		for filt in filter_data:                       
			median = dat_[filt].median()
			dat_[filt].fillna(median, inplace = True)        
		datas = dat_[filter_info][:].values
		del_list = np.where(dat_['volume'] == 0)
		datas = np.delete(datas, del_list, axis=0) 
		## handle the exception end
		
		rows = len(datas)
		if rows <= (hold_days + 1):
			continue        
		## tranform the data  price = price / pre_close_price for the ones of open close high low
		pre_close = datas[0:1, 3:4][0][0]        
		for i in range(1, rows) :
			## price record
			if (i < rows -hold_days - predict_days):
				Label_price = datas[i+hold_days:i+hold_days+1,1:2][0][0]
				tommorow_prices.append(Label_price)
				Label_price = datas[i+hold_days-1:i+hold_days,1:2][0][0]                
				today_prices.append(Label_price)
			## reform the price           
			pre_close_hold = datas[i:i+1, 3:4][0][0]            
			datas[i:i+1, 0:4] = (datas[i:i+1, 0:4] - pre_close) / pre_close
			pre_close = pre_close_hold    
			#print(datas[i:i+1, 0:4])
    
		## handle the labels  and the last predict_days of pc_changes ,if more than 2 (2%) as postiave 1 label others as 0   
		each_labels_num = rows - hold_days - predict_days
		for i in range(1, each_labels_num) :
			## train data  100000 for solveing forcas the error "OverflowError: Python int too large to convert to C long"
			mean_ = np.mean(datas[i:i+hold_days, 4:6]/100000, axis=0)
			std_ = np.std((datas[i:i+hold_days, 4:6]/100000).astype(np.int32), axis=0)
			if std_[0:1] == 0 or std_[1:2] == 0:
				continue           
			datas[i:i+hold_days, 4:6] = (datas[i:i+hold_days, 4:6]/100000 - mean_) / std_              
			each = datas[i:i + hold_days,:]
			each[each==np.nan] = 0                        
			## normalize the volue and amount
			group.append(each[:,0:7])
			
			## labels           
			ratio_record = datas[i+hold_days:i+hold_days+predict_days, 7:8]
			#print(ratio_record)
			sum_all = np.sum(ratio_record, axis=0)
			#print(sum_all)            
			if sum_all > gain_threshold :
				Label = 1
			else :
				Label = 0 
			labels.append(Label)
			## info record
			code = datas[i+hold_days:i+hold_days+1,8:9]  
			stock_code.append(code)
	#print(group)            
	return np.array(group), np.array(labels), np.array(tommorow_prices), np.array(today_prices), np.array(stock_code)


def read_stock_data_items_and_transform_by_path(data_path, n_dim_feautures = 12, hold_days = 5, predict_days = 5, gain_threshold = 2.0):
	""" Read data """
	# Fixed params
	n_predict = 2
	filter_data = ['open','high','low','close','volume','amount','turn','pctChg']   
	filter_info = ['open','high','low','close','volume','amount','turn','pctChg','code'] 
	# Read time-series data    
	file_lists = []
	file_lists.append(data_path)
	group = [] 
	labels = []
	tommorow_prices = []  
	today_prices = []    
	stock_code = []     
	each = np.zeros((1, hold_days, n_dim_feautures))
	Label = np.zeros(n_predict) 
	for csv in file_lists:
		dat_ = pd.read_csv(csv)
		
		## handle the exception
		for filt in filter_data:                       
			median = dat_[filt].median()
			dat_[filt].fillna(median, inplace = True)        
		datas = dat_[filter_info][:].values
		del_list = np.where(dat_['volume'] == 0)
		datas = np.delete(datas, del_list, axis=0) 
		## handle the exception end
		
		rows = len(datas)
		if rows <= (hold_days + 1):
			continue        
		## tranform the data  price = price / pre_close_price for the ones of open close high low
		pre_close = datas[0:1, 3:4][0][0]        
		for i in range(1, rows) :
			## price record
			if (i < rows -hold_days - predict_days):
				Label_price = datas[i+hold_days:i+hold_days+1,1:2][0][0]
				tommorow_prices.append(Label_price)
				Label_price = datas[i+hold_days-1:i+hold_days,1:2][0][0]                
				today_prices.append(Label_price)
			## reform the price           
			pre_close_hold = datas[i:i+1, 3:4][0][0]            
			datas[i:i+1, 0:4] = (datas[i:i+1, 0:4] - pre_close) / pre_close
			pre_close = pre_close_hold    
			#print(datas[i:i+1, 0:4])
    
		## handle the labels  and the last predict_days of pc_changes ,if more than 2 (2%) as postiave 1 label others as 0   
		each_labels_num = rows - hold_days - predict_days
		for i in range(1, each_labels_num) :
			## train data  100000 for solveing forcas the error "OverflowError: Python int too large to convert to C long"
			mean_ = np.mean(datas[i:i+hold_days, 4:6]/100000, axis=0)
			std_ = np.std((datas[i:i+hold_days, 4:6]/100000).astype(np.int32), axis=0)
			if std_[0:1] == 0 or std_[1:2] == 0:
				continue            
			datas[i:i+hold_days, 4:6] = (datas[i:i+hold_days, 4:6]/100000 - mean_) / std_              
			each = datas[i:i + hold_days,:]
			each[each==np.nan] = 0                        
			## normalize the volue and amount
			group.append(each[:,0:7])
			
			## labels           
			ratio_record = datas[i+hold_days:i+hold_days+predict_days, 7:8]
			#print(ratio_record)
			sum_all = np.sum(ratio_record, axis=0)
			#print(sum_all)            
			if sum_all > gain_threshold :
				Label = 1
			else :
				Label = 0 
			labels.append(Label)
			## info record
			code = datas[i+hold_days:i+hold_days+1,8:9]  
			stock_code.append(code)
	#print(group)            
	return np.array(group), np.array(labels), np.array(tommorow_prices), np.array(today_prices), np.array(stock_code)

def read_stock_data_items_and_transform_with_some_days(data_path, n_dim_feautures = 12, hold_days = 5, predict_days = 5, gain_threshold = 2.0):
	""" Read data """
	# Fixed params
	n_predict = 2
	filter_data = ['open','high','low','close','volume','amount','turn','pctChg']   
	filter_info = ['open','high','low','close','volume','amount','turn','pctChg','code'] 
	# Read time-series data    
	file_lists = []
	file_lists.append(data_path)
	group = [] 
	today_price = 0     
	stock_code = []     
	each = np.zeros((1, hold_days, n_dim_feautures))
	Label = np.zeros(n_predict) 
	for csv in file_lists:
		dat_ = pd.read_csv(csv)
		
		## handle the exception
		for filt in filter_data:                       
			median = dat_[filt].median()
			dat_[filt].fillna(median, inplace = True)        
		datas = dat_[filter_info][:].values
		del_list = np.where(dat_['volume'] == 0)
		datas = np.delete(datas, del_list, axis=0) 
		## handle the exception end
		
		rows = len(datas)
		if rows <= (hold_days + 1):
			continue        
		## tranform the data  price = price / pre_close_price for the ones of open close high low
		pre_close = datas[rows-hold_days-1:rows-hold_days, 3:4][0][0] 
		#print(pre_close)
		for i in range(rows-hold_days-1, rows) :
			## record price            
			today_price = datas[rows-hold_days-1:rows-hold_days,1:2][0][0]
			            
			## reform the price           
			pre_close_hold = datas[i:i+1, 3:4][0][0]            
			datas[i:i+1, 0:4] = (datas[i:i+1, 0:4] - pre_close) / pre_close
			pre_close = pre_close_hold    
			#print(datas[i:i+1, 0:4])
    
		## handle the labels  and the last predict_days of pc_changes ,if more than 2 (2%) as postiave 1 label others as 0   
		each_labels_num = rows - hold_days
		#print(each_labels_num)       
		for i in range(each_labels_num, each_labels_num+1) :
			#print(datas[i:i+hold_days, 4:6])
			## train data  100000 for solveing forcas the error "OverflowError: Python int too large to convert to C long"
			mean_ = np.mean(datas[i:i+hold_days, 4:6]/1000000, axis=0)
			std_ = np.std((datas[i:i+hold_days, 4:6]/1000000).astype(np.int32), axis=0)
			if std_[0:1] == 0 or std_[1:2] == 0:
				continue            
			datas[i:i+hold_days, 4:6] = (datas[i:i+hold_days, 4:6]/1000000 - mean_) / std_              
			each = datas[i:i + hold_days,:]
			each[each==np.nan] = 0                        
			## normalize the volue and amount
			group.append(each[:,0:7])
			
			## labels           
			ratio_record = datas[i+hold_days:i+hold_days+predict_days, 7:8]
			#print(ratio_record)
			sum_all = np.sum(ratio_record, axis=0)
			## info record
			code = datas[i+hold_days:i+hold_days+1,8:9]  
			stock_code.append(code)
	#print(group)            
	return np.array(group), today_price, np.array(stock_code)

def standardize(train, test):
	""" Standardize data """
	
	X_train=[]
	X_test = []
	# Standardize train and test
	    
	for each in train :
		print(each.shape)
		each = (each - np.mean(each, axis=0)) / np.std(each, axis=0)
		X_train.append(each)
	for each2 in test :
		print(each)
		each2 = (each2 - np.mean(each2, axis=0)) / np.std(each2, axis=0)
		X_test.append(each2)
	        
	#X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
	#X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]
	return np.array(X_train), np.array(X_test)

def one_hot(labels, n_class = 6):
	""" One-hot encoding """
	expansion = np.eye(n_class)
	y = expansion[:, labels-1].T
	assert y.shape[1] == n_class, "Wrong number of labels!"

	return y

def get_batches(X, y, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size]
	
def get_batches_exp(X, y, z, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y, z = X[:n_batches*batch_size], y[:n_batches*batch_size], z[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size], z[b:b+batch_size]    

def get_batches_exp_2(X, y, z, v, batch_size = 100):
	""" Return a generator for batches """
	n_batches = len(X) // batch_size
	X, y, z, v = X[:n_batches*batch_size], y[:n_batches*batch_size], z[:n_batches*batch_size], v[:n_batches*batch_size]

	# Loop over batches and yield
	for b in range(0, len(X), batch_size):
		yield X[b:b+batch_size], y[b:b+batch_size], z[b:b+batch_size] , v[b:b+batch_size] 


