# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 20:16:30 2015

@author: melgazar9
"""

# Read in all CME data from EODData.com
import pandas as pd
import numpy as np
import os

main_path = '/Users/melgazar9/Desktop/CME_Data'

CME_2005_Files = [i for i in os.listdir(main_path + '/' + 'CME_2005')]
CME_2006_Files = [i for i in os.listdir(main_path + '/' + 'CME_2006')]
CME_2007_Files = [i for i in os.listdir(main_path + '/' + 'CME_2007')]
CME_2008_Files = [i for i in os.listdir(main_path + '/' + 'CME_2008')]
CME_2009_Files = [i for i in os.listdir(main_path + '/' + 'CME_2009')]
CME_2010_Files = [i for i in os.listdir(main_path + '/' + 'CME_2010')]
CME_2011_Files = [i for i in os.listdir(main_path + '/' + 'CME_2011')]
CME_2012_Files = [i for i in os.listdir(main_path + '/' + 'CME_2012')]
CME_2013_Files = [i for i in os.listdir(main_path + '/' + 'CME_2013')]
CME_2014_Files = [i for i in os.listdir(main_path + '/' + 'CME_2014')]
CME_2015_Files = [i for i in os.listdir(main_path + '/' + 'CME_2015')]
CME_2005_Data = pd.DataFrame()
CME_2006_Data = pd.DataFrame()
CME_2007_Data = pd.DataFrame()
CME_2008_Data = pd.DataFrame()
CME_2009_Data = pd.DataFrame()
CME_2010_Data = pd.DataFrame()
CME_2011_Data = pd.DataFrame()
CME_2012_Data = pd.DataFrame()
CME_2013_Data = pd.DataFrame()
CME_2014_Data = pd.DataFrame()
CME_2015_Data = pd.DataFrame()

# Read in CME_2005_Data
count = 1
while count < len(CME_2005_Files):
    #print CME_2005_Files[count]
    #count += 1
    CME_2005_Data = pd.concat([CME_2005_Data, pd.read_csv(main_path + '/' + 'CME_2005' + '/' + CME_2005_Files[count])])
    count += 1
#print CME_2005_Data
CME_2005_Data.to_csv(main_path + '/' + 'CME_2005_Data.txt')


# Read in CME_2006_Data
count = 1
while count < len(CME_2006_Files):
    #print CME_2006_Files[count]
    #count += 1
    CME_2006_Data = pd.concat([CME_2006_Data, pd.read_csv(main_path + '/' + 'CME_2006' + '/' + CME_2006_Files[count])])
    count += 1
#print CME_2006_Data
CME_2006_Data.to_csv(main_path + '/' + 'CME_2006_Data.txt')


# Read in CME_2007_Data
count = 1
while count < len(CME_2007_Files):
    #print CME_2007_Files[count]
    #count += 1
    CME_2007_Data = pd.concat([CME_2007_Data, pd.read_csv(main_path + '/' + 'CME_2007' + '/' + CME_2007_Files[count])])
    count += 1
#print CME_2007_Data
CME_2007_Data.to_csv(main_path + '/' + 'CME_2007_Data.txt')


# Read in CME_2008_Data
count = 1
while count < len(CME_2008_Files):
    #print CME_2008_Files[count]
    #count += 1
    CME_2008_Data = pd.concat([CME_2008_Data, pd.read_csv(main_path + '/' + 'CME_2008' + '/' + CME_2008_Files[count])])
    count += 1
#print CME_2008_Data
CME_2008_Data.to_csv(main_path + '/' + 'CME_2008_Data.txt')


# Read in CME_2009_Data
count = 1
while count < len(CME_2009_Files):
    #print CME_2009_Files[count]
    #count += 1
    CME_2009_Data = pd.concat([CME_2009_Data, pd.read_csv(main_path + '/' + 'CME_2009' + '/' + CME_2009_Files[count])])
    count += 1
#print CME_2009_Data
CME_2009_Data.to_csv(main_path + '/' + 'CME_2009_Data.txt')


# Read in CME_2010_Data
count = 1
while count < len(CME_2010_Files):
    #print CME_2010_Files[count]
    #count += 1
    CME_2010_Data = pd.concat([CME_2010_Data, pd.read_csv(main_path + '/' + 'CME_2010' + '/' + CME_2010_Files[count])])
    count += 1
#print CME_2010_Data
CME_2010_Data.to_csv(main_path + '/' + 'CME_2010_Data.txt')


# Read in CME_2011_Data
count = 1
while count < len(CME_2011_Files):
    #print CME_2011_Files[count]
    #count += 1
    CME_2011_Data = pd.concat([CME_2011_Data, pd.read_csv(main_path + '/' + 'CME_2011' + '/' + CME_2011_Files[count])])
    count += 1
#print CME_2011_Data
CME_2011_Data.to_csv(main_path + '/' + 'CME_2011_Data.txt')


# Read in CME_2012_Data
count = 1
while count < len(CME_2012_Files):
    #print CME_2012_Files[count]
    #count += 1
    CME_2012_Data = pd.concat([CME_2012_Data, pd.read_csv(main_path + '/' + 'CME_2012' + '/' + CME_2012_Files[count])])
    count += 1
#print CME_2012_Data
CME_2012_Data.to_csv(main_path + '/' + 'CME_2012_Data.txt')


# Read in CME_2013_Data
count = 1
while count < len(CME_2013_Files):
    #print CME_2013_Files[count]
    #count += 1
    CME_2013_Data = pd.concat([CME_2013_Data, pd.read_csv(main_path + '/' + 'CME_2013' + '/' + CME_2013_Files[count])])
    count += 1
#print CME_2013_Data
CME_2013_Data.to_csv(main_path + '/' + 'CME_2013_Data.txt')


# Read in CME_2014_Data
count = 1
while count < len(CME_2014_Files):
    #print CME_2014_Files[count]
    #count += 1
    CME_2014_Data = pd.concat([CME_2014_Data, pd.read_csv(main_path + '/' + 'CME_2014' + '/' + CME_2014_Files[count])])
    count += 1
#print CME_2014_Data
CME_2014_Data.to_csv(main_path + '/' + 'CME_2014_Data.txt')


# Read in CME_2015_Data
count = 1
while count < len(CME_2015_Files):
    #print CME_2015_Files[count]
    #count += 1
    CME_2015_Data = pd.concat([CME_2015_Data, pd.read_csv(main_path + '/' + 'CME_2015' + '/' + CME_2015_Files[count])])
    count += 1
#print CME_2015_Data
CME_2015_Data.to_csv(main_path + '/' + 'CME_2015_Data.txt')


CME_All_Text_Files = [i for i in os.listdir(main_path) if i[-4:] == '.txt']
CME_OHLC_2005_to_2015 = pd.DataFrame()

count = 0
CME_2005_All_Data = pd.read_csv(main_path + '/' + 'CME_2005_Data.txt')
count = 0
CME_2005_All_Data = pd.DataFrame()
while count < len(CME_All_Text_Files):
    #print CME_2015_Files[count]
    #count += 1
    CME_OHLC_2005_to_2015 = pd.concat([CME_2005_All_Data, pd.read_csv(main_path + '/' + CME_All_Text_Files[count])])
    count += 1
#print CME_2015_Data
CME_OHLC_2005_to_2015.to_csv(main_path + '/' + 'CME_OHLC_2005_to_2015.txt')


.

nymex_path = '/Users/melgazar9/Desktop/NYMEX_Data'

NYMEX_2012_Files = [i for i in os.listdir(nymex_path + '/' + 'NYMEX_2012')]
NYMEX_2013_Files = [i for i in os.listdir(nymex_path + '/' + 'NYMEX_2013')]
NYMEX_2014_Files = [i for i in os.listdir(nymex_path + '/' + 'NYMEX_2014')]
NYMEX_2015_Files = [i for i in os.listdir(nymex_path + '/' + 'NYMEX_2015')]
NYMEX_2012_Data = pd.DataFrame()
NYMEX_2013_Data = pd.DataFrame()
NYMEX_2014_Data = pd.DataFrame()
NYMEX_2015_Data = pd.DataFrame()



# Read in NYMEX_2012_Data
count = 1
while count < len(NYMEX_2012_Files):
    #print NYMEX_2012_Files[count]
    #count += 1
    NYMEX_2012_Data = pd.concat([NYMEX_2012_Data, pd.read_csv(nymex_path + '/' + 'NYMEX_2012' + '/' + NYMEX_2012_Files[count])])
    count += 1
#print NYMEX_2012_Data
NYMEX_2012_Data.to_csv(nymex_path + '/' + 'NYMEX_2012_Data.txt')
CL_Data_2012 = NYMEX_2012_Data.loc[NYMEX_2012_Data['<ticker>'] == 'CL']
CL_Data_2012.to_csv(nymex_path + '/' + 'NYMEX2012_Data.csv')


# Read in NYMEX_2013_Data
count = 1
while count < len(NYMEX_2013_Files):
    #print NYMEX_2013_Files[count]
    #count += 1
    NYMEX_2013_Data = pd.concat([NYMEX_2013_Data, pd.read_csv(nymex_path + '/' + 'NYMEX_2013' + '/' + NYMEX_2013_Files[count])])
    count += 1
#print NYMEX_2013_Data
NYMEX_2013_Data.to_csv(nymex_path + '/' + 'NYMEX_2013_Data.csv')
CL_Data_2013 = NYMEX_2013_Data.loc[NYMEX_2013_Data['<ticker>'] == 'CL']
CL_Data_2013.to_csv(nymex_path + '/' + 'NYMEX2013_Data.csv')


# Read in NYMEX_2014_Data
count = 1
while count < len(NYMEX_2014_Files):
    #print NYMEX_2014_Files[count]
    #count += 1
    NYMEX_2014_Data = pd.concat([NYMEX_2014_Data, pd.read_csv(nymex_path + '/' + 'NYMEX_2014' + '/' + NYMEX_2014_Files[count])])
    count += 1
#print NYMEX_2014_Data
NYMEX_2014_Data.to_csv(nymex_path + '/' + 'NYMEX_2014_Data.csv')
CL_Data_2014 = NYMEX_2014_Data.loc[NYMEX_2014_Data['<ticker>'] == 'CL']
CL_Data_2014.to_csv(nymex_path + '/' + 'NYMEX2014_Data.csv')


# Read in NYMEX_2015_Data
count = 1
while count < len(NYMEX_2015_Files):
    #print NYMEX_2015_Files[count]
    #count += 1
    NYMEX_2015_Data = pd.concat([NYMEX_2015_Data, pd.read_csv(nymex_path + '/' + 'NYMEX_2015' + '/' + NYMEX_2015_Files[count])])
    count += 1
#print NYMEX_2015_Data
NYMEX_2015_Data.to_csv(nymex_path + '/' + 'NYMEX_2015_Data.csv')
CL_Data_2015 = NYMEX_2015_Data.loc[NYMEX_2015_Data['<ticker>'] == 'CL']
CL_Data_2015.to_csv(nymex_path + '/' + 'NYMEX2015_Data.csv')