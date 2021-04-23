import numpy as np
from flask import Flask, request, jsonify, render_template, json
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import datetime
from datetime import datetime
from datetime import timezone

application = Flask(__name__)

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
geolocator = Nominatim(user_agent="modelling-Crime_Uni")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

#print(location.raw['address']['postcode'])
@application.route('/')
def home():
    return render_template('index.html')

@application.route('/index.html')
def home2():
    return render_template('index.html')

@application.route('/Compare.html')
def ComparePage():
    return render_template('Compare.html')

@application.route('/RegionalTrends.html')
def RegionalTrendsPage():
    return render_template('RegionalTrends.html')
    
@application.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method == 'POST':
        ListPostcode = list(request.form.values())
        global PostcodeInput
        global firstMonthInput 
        global firstYearInput
        global secondMonthInput
        global secondYearInput
        
        firstMonthInput = request.form.get('firstmonthDropDown')
        firstYearInput = request.form.get('firstyearDropDown')
        
        secondMonthInput = request.form.get('secoondmonthDropDown')
        secondYearInput = request.form.get('secondyearDropDown')
        
        PostcodeInput = ListPostcode[0]
        print(PostcodeInput)
        print(firstMonthInput)
        print(firstYearInput)
        print(secondMonthInput)
        print(secondYearInput)
        #location = PostcodeInput.apply(geocode)
        global location
        location = geolocator.geocode(PostcodeInput)
        #prediction = model.predict(final_features)'
        return render_template('submitted.html')
   # return '{} {} {}'.format((location.latitude, location.longitude)
    if request.method == 'GET':
        addressVariable = geolocator.reverse((location.latitude,location.longitude), timeout=None)
        print(addressVariable)
        postcodeVariable = addressVariable.raw['address']['postcode']
        print(addressVariable.raw['address']['county'])
        cityVariable = ''
        try:
            countyVariable = addressVariable.raw['address']['county']
        except:
            cityVariable = addressVariable.raw['address']['city']
            
        if (cityVariable == 'Plymouth'):
            countyVariable = 'Devon'
        elif (cityVariable == 'City of London'):
            countyVariable = 'City of London'
        
        OutwardPostcodeVariable = postcodeVariable.split(' ')
        OutwardPostcodeVariable = OutwardPostcodeVariable[0]
        
        if ((countyVariable == 'Cornwall') or (countyVariable == 'Devon')):
            dataset = pd.read_csv('DEVONCORNWALL.csv', low_memory=False);
        elif (countyVariable == 'City of London'):
            dataset = pd.read_csv('CITY_OF_LONDON.csv', low_memory=False);
        elif (countyVariable == 'Dorset') or (countyVariable == 'Bournemouth, Christchurch and Poole'):
            dataset = pd.read_csv('DORSET.csv', low_memory=False);
        elif (countyVariable == 'Surrey'):
            dataset = pd.read_csv('SURREY.csv', low_memory=False);
        elif (countyVariable == 'North Yorkshire'):
            dataset = pd.read_csv('NORTHYORKSHIRE.csv', low_memory=False);
        elif (countyVariable == 'York'):
            dataset = pd.read_csv('NORTHYORKSHIRE.csv', low_memory=False);
        elif (countyVariable == 'Wiltshire'):
            dataset = pd.read_csv('WILTSHIRE.csv', low_memory=False);   
        elif (countyVariable == 'Swindon'):
            dataset = pd.read_csv('WILTSHIRE.csv', low_memory=False);   
        elif (countyVariable == 'Buckinghamshire'):
            dataset = pd.read_csv('THAMES_VALLEY.csv', low_memory=False); 
        elif (countyVariable == 'Oxfordshire'):
            dataset = pd.read_csv('THAMES_VALLEY.csv', low_memory=False);  
        elif (countyVariable == 'Bracknell Forest'):
            dataset = pd.read_csv('THAMES_VALLEY.csv', low_memory=False);  
        elif (countyVariable == 'West Berkshire'):
            dataset = pd.read_csv('THAMES_VALLEY.csv', low_memory=False);
        elif (countyVariable == 'Reading'):
            dataset = pd.read_csv('THAMES_VALLEY.csv', low_memory=False);
        elif (countyVariable == 'Wokingham'):
            dataset = pd.read_csv('THAMES_VALLEY.csv', low_memory=False);
            
            
        dt = datetime(int(firstYearInput), int(firstMonthInput), 1)
        timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
        
        dt = datetime(int(secondYearInput), int(secondMonthInput), 1)
        timestamp2 = dt.replace(tzinfo=timezone.utc).timestamp()
        
        PlottingDataset = dataset[(dataset['Month in Unix'] >= timestamp) & (dataset['Month in Unix'] <= timestamp2)]
        
        
        SmallestUnixValue = PlottingDataset['Month in Unix'].min()
        
        LargestUnixValue = PlottingDataset['Month in Unix'].max()
        
        SmallestDate = (datetime.utcfromtimestamp(SmallestUnixValue).strftime('%m/%Y'))
        LargestDate = (datetime.utcfromtimestamp(LargestUnixValue).strftime('%m/%Y'))
        
        NextMonthValue = LargestUnixValue + 2592000

        
        InwardDataset = PlottingDataset

        PlottingDataset = PlottingDataset.loc[PlottingDataset['Outward Postcode'] == OutwardPostcodeVariable]
        
        InwardDataset = InwardDataset.loc[InwardDataset['Postcode'] == postcodeVariable]
        #print(PlottingDataset)
        PlottingDataset['MAPLongitude'] = 0.0

        PlottingDataset['MAPLatitude'] = 0.0
        PlottingDataset['SeverityRating'] = 0.0
        
        NumberInwardCrimes = InwardDataset.shape[0]
        NumberInwardCrimesOutward = PlottingDataset.shape[0]

        
        modeDatasetInward = InwardDataset
        modeDatasetInward = modeDatasetInward.mode()
        modeDatasetInward = modeDatasetInward[['Crime type']].copy()

        
        PlottingDataset['MAPLongitude'] = location.longitude
        PlottingDataset['MAPLatitude'] = location.latitude
        #print(PlottingDataset)

        #print(jsonPlottingValues)
        modeDataset = PlottingDataset
        modeDataset = modeDataset.mode()
        modeDataset = modeDataset[['Crime type']].copy()
        print(modeDataset['Crime type'].values[0])
        
        ## SEVERITY RATING MODEL
        dataset = dataset.loc[dataset['Outward Postcode'] == OutwardPostcodeVariable]
        dataset = dataset[['Longitude', 'Latitude', 'Severity', 'Month Number', 'Month in Unix', 'Year Number']].copy()
        X = dataset[['Month Number', 'Month in Unix', 'Year Number']]
        y =  dataset[['Severity']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        y_pred = model.predict(X_test)
                               
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        print('Root Mean squared error: %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
        
        
        ##NUMBER OF CRIMES MODELLING
        NumberOfCrimes = dataset[['Month in Unix']].copy()
        NumberOfCrimes['Count'] = 0
        NumberOfCrimes['Count']=NumberOfCrimes.groupby(by='Month in Unix')['Month in Unix'].transform('count')
        NumberOfCrimes = NumberOfCrimes.drop_duplicates()
        XNumberofCrimes = NumberOfCrimes[['Month in Unix']]
        yNumberofCrimes =  NumberOfCrimes[['Count']]

        X_trainNumberofCrimes, X_testNumberofCrimes, y_trainNumberofCrimes, y_testNumberofCrimes = train_test_split(XNumberofCrimes, yNumberofCrimes, test_size=0.1, random_state=42)
        modelNumberofCrimes = LinearRegression()
        modelNumberofCrimes.fit(X_trainNumberofCrimes, y_trainNumberofCrimes)
        modelNumberofCrimes.score(X_testNumberofCrimes, y_testNumberofCrimes)
        y_predNumberofCrimes = modelNumberofCrimes.predict(X_testNumberofCrimes)

        print('Mean squared error: %.2f' % mean_squared_error(y_testNumberofCrimes, y_predNumberofCrimes))
        print('Root Mean squared error: %.2f' % np.sqrt(mean_squared_error(y_testNumberofCrimes, y_predNumberofCrimes)))
        print('Coefficient of determination: %.2f' % r2_score(y_testNumberofCrimes, y_predNumberofCrimes))

        NewMonthNumber = (datetime.utcfromtimestamp(NextMonthValue).strftime('%m'))
        NewYearNumber = (datetime.utcfromtimestamp(NextMonthValue).strftime('%Y'))
        

        severityRating = model.predict([[NewMonthNumber, NextMonthValue, NewYearNumber]])
        NumberOfCrimePrediction = modelNumberofCrimes.predict([[NextMonthValue]])
        
        if NumberOfCrimePrediction[0][0] < 0 :
            NumberOfCrimePrediction[0][0] = 0
            
        print(severityRating[0][0])
        print(NumberOfCrimePrediction[0][0])
        
        message = {'latitudeHead':location.latitude, 'longitudeHead':location.longitude}
        crimeIncreasingDecreasingVar = ""
        print('-----Model-----')
        print('intercept:', model.intercept_)

        print('slope:', model.coef_)
        print('-----ModelofCrimes-----')
        print('intercept:', modelNumberofCrimes.intercept_)
        print('slope:', modelNumberofCrimes.coef_)
        if modelNumberofCrimes.coef_ < 0:
            print('Decreasing')
            crimeIncreasingDecreasingVar = "Decreasing"
        else:
            print('Increasing')
            crimeIncreasingDecreasingVar = "Increasing"
        print(message)
        #message = {'Longitude':location.longitude, "latitude":location.latitude}
        #message = jsonify(message)
        #message2 = jsonify(jsonPlottingValues)
        #message = message + message2
        PlottingDataset['SeverityRating'] = round(severityRating[0][0], 2)
        PlottingDataset['PredictedNumberOfCrimes'] = round(NumberOfCrimePrediction[0][0], 0)
        PlottingDataset['ModeCrime'] = modeDataset['Crime type'].values[0]
        PlottingDataset['inputPostcode'] = postcodeVariable
        
        PlottingDataset['NumberInwardCrimes'] = NumberInwardCrimes
        
        try:
            PlottingDataset['ModeCrimeInward'] = modeDatasetInward['Crime type'].values[0]
        except:
            PlottingDataset['ModeCrimeInward'] = ""

        PlottingDataset['NumberOutwardCrimes'] = NumberInwardCrimesOutward

        PlottingDataset['FirstMonth'] = SmallestDate
        
        PlottingDataset['SecondMonth'] = LargestDate

        PlottingDataset['CrimeIncreasingDecreasing'] = crimeIncreasingDecreasingVar

        jsonPlottingValues = PlottingDataset.to_json(orient="records")
        return jsonPlottingValues  # serialize and use JSON headers                       
                             
@application.route('/compare',methods=['GET','POST'])
def compare():
    '''
    For rendering results on HTML GUI
    '''
    
    if request.method == 'POST':
        ListPostcode = list(request.form.values())
        global PostcodeInput1
        global PostcodeInput2
        global firstMonthInput 
        global firstYearInput
        global secondMonthInput
        global secondYearInput
        
        firstMonthInput = request.form.get('firstmonthDropDown')
        firstYearInput = request.form.get('firstyearDropDown')
        
        secondMonthInput = request.form.get('secoondmonthDropDown')
        secondYearInput = request.form.get('secondyearDropDown')
        
        PostcodeInput1 = ListPostcode[0]
        PostcodeInput2 = ListPostcode[1]
        print(PostcodeInput1)
        print(PostcodeInput2)
        print(firstMonthInput)
        print(firstYearInput)
        print(secondMonthInput)
        print(secondYearInput)
        #location = PostcodeInput.apply(geocode)
        global location1
        global location2
        location1 = geolocator.geocode(PostcodeInput1)
        location2 = geolocator.geocode(PostcodeInput2)
        #prediction = model.predict(final_features)'
        return render_template('submittedCompare.html', longitude_text='{}'.format(location1.longitude), latitude_text='{}'.format(location1.latitude))
   # return '{} {} {}'.format((location.latitude, location.longitude)
    if request.method == 'GET':
        addressVariable = geolocator.reverse((location1.latitude,location1.longitude), timeout=None)
        addressVariable2 = geolocator.reverse((location2.latitude,location2.longitude), timeout=None)
        
        postcodeVariable = addressVariable.raw['address']['postcode']
        postcodeVariable2 = addressVariable2.raw['address']['postcode']
        
        cityVariable = ''
        cityVariable2 = ''
        
        try:
            countyVariable = addressVariable.raw['address']['county']
            countyVariable2 = addressVariable2.raw['address']['county']
        except:
            cityVariable = addressVariable.raw['address']['city']
            cityVariable2 = addressVariable2.raw['address']['city']
            
        if (cityVariable == 'Plymouth'):
            countyVariable = 'Devon'
        elif (cityVariable == 'City of London'):
            countyVariable = 'City of London'
            
        if (cityVariable2 == 'Plymouth'):
            countyVariable2 = 'Devon'
        elif (cityVariable2 == 'City of London'):
            countyVariable2 = 'City of London'
        
        OutwardPostcodeVariable = postcodeVariable.split(' ')
        OutwardPostcodeVariable = OutwardPostcodeVariable[0]
        
        OutwardPostcodeVariable2 = postcodeVariable2.split(' ')
        OutwardPostcodeVariable2 = OutwardPostcodeVariable2[0]
        
        if ((countyVariable == 'Cornwall') or (countyVariable == 'Devon')):
            dataset = pd.read_csv('DEVONCORNWALL.csv', low_memory=False);
        elif (countyVariable == 'City of London'):
            dataset = pd.read_csv('CITY_OF_LONDON.csv', low_memory=False);
        elif (countyVariable == 'Dorset') or (countyVariable == 'Bournemouth, Christchurch and Poole'):
            dataset = pd.read_csv('DORSET.csv', low_memory=False);
        elif (countyVariable == 'Surrey'):
            dataset = pd.read_csv('SURREY.csv', low_memory=False);
        elif (countyVariable == 'North Yorkshire'):
            dataset = pd.read_csv('NORTHYORKSHIRE.csv', low_memory=False);
        elif (countyVariable == 'York'):
            dataset = pd.read_csv('NORTHYORKSHIRE.csv', low_memory=False);
            
        if ((countyVariable2 == 'Cornwall') or (countyVariable2 == 'Devon')):
            dataset2 = pd.read_csv('DEVONCORNWALL.csv', low_memory=False);
        elif (countyVariable2 == 'City of London'):
            dataset2 = pd.read_csv('CITY_OF_LONDON.csv', low_memory=False);
        elif (countyVariable2 == 'Dorset') or (countyVariable2 == 'Bournemouth, Christchurch and Poole'):
            dataset2 = pd.read_csv('DORSET.csv', low_memory=False);
        elif (countyVariable2 == 'Surrey'):
            dataset2 = pd.read_csv('SURREY.csv', low_memory=False);
        elif (countyVariable2 == 'North Yorkshire'):
            dataset2 = pd.read_csv('NORTHYORKSHIRE.csv', low_memory=False);
        elif (countyVariable2 == 'York'):
            dataset2 = pd.read_csv('NORTHYORKSHIRE.csv', low_memory=False);
            
        dt = datetime(int(firstYearInput), int(firstMonthInput), 1)
        timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
        
        dt = datetime(int(secondYearInput), int(secondMonthInput), 1)
        timestamp2 = dt.replace(tzinfo=timezone.utc).timestamp()
        
        PlottingDataset = dataset[(dataset['Month in Unix'] >= timestamp) & (dataset['Month in Unix'] <= timestamp2)]
        PlottingDataset2 = dataset2[(dataset2['Month in Unix'] >= timestamp) & (dataset2['Month in Unix'] <= timestamp2)]

        SmallestUnixValue = PlottingDataset['Month in Unix'].min()
        LargestUnixValue = PlottingDataset['Month in Unix'].max()
        
        NextMonthValue = LargestUnixValue + 2592000
        
        SmallestDate = (datetime.utcfromtimestamp(SmallestUnixValue).strftime('%m/%Y'))
        LargestDate = (datetime.utcfromtimestamp(LargestUnixValue).strftime('%m/%Y'))
        
        SmallestUnixValue2 = PlottingDataset2['Month in Unix'].min()
        LargestUnixValue2 = PlottingDataset2['Month in Unix'].max()
        
        NextMonthValue2 = LargestUnixValue2 + 2592000
        
        SmallestDate2 = (datetime.utcfromtimestamp(SmallestUnixValue2).strftime('%m/%Y'))
        LargestDate2 = (datetime.utcfromtimestamp(LargestUnixValue2).strftime('%m/%Y'))

        InwardDataset = PlottingDataset
        InwardDataset2 = PlottingDataset2

        PlottingDataset = PlottingDataset.loc[PlottingDataset['Outward Postcode'] == OutwardPostcodeVariable]
        PlottingDataset2 = PlottingDataset2.loc[PlottingDataset2['Outward Postcode'] == OutwardPostcodeVariable2]
        
        InwardDataset = InwardDataset.loc[InwardDataset['Postcode'] == postcodeVariable]
        InwardDataset2 = InwardDataset2.loc[InwardDataset2['Postcode'] == postcodeVariable2]
        
        NumberInwardCrimes = InwardDataset.shape[0]
        NumberInwardCrimes2 = InwardDataset2.shape[0]
        
        NumberInwardCrimesOutward = PlottingDataset.shape[0]
        NumberInwardCrimesOutward2 = PlottingDataset2.shape[0]

        modeDatasetInward = InwardDataset
        modeDatasetInward = modeDatasetInward.mode()
        modeDatasetInward = modeDatasetInward[['Crime type']].copy()
        
        modeDatasetInward2 = InwardDataset2
        modeDatasetInward2 = modeDatasetInward2.mode()
        modeDatasetInward2 = modeDatasetInward2[['Crime type']].copy()
        
        #print(PlottingDataset)
        PlottingDataset['MAPLongitude'] = 0.0
        PlottingDataset2['MAPLongitude'] = 0.0

        PlottingDataset['MAPLatitude'] = 0.0
        PlottingDataset2['MAPLatitude'] = 0.0
        
        PlottingDataset['SeverityRating'] = 0.0
        PlottingDataset2['SeverityRating'] = 0.0

        
        PlottingDataset['MAPLongitude'] = location1.longitude
        PlottingDataset['MAPLatitude'] = location1.latitude
        
        PlottingDataset2['MAPLongitude'] = location2.longitude
        PlottingDataset2['MAPLatitude'] = location2.latitude
        #print(PlottingDataset)

        #print(jsonPlottingValues)
        modeDataset = PlottingDataset
        modeDataset = modeDataset.mode()
        modeDataset = modeDataset[['Crime type']].copy()
        print(modeDataset['Crime type'].values[0])
        
        modeDataset2 = PlottingDataset2
        modeDataset2 = modeDataset2.mode()
        modeDataset2 = modeDataset2[['Crime type']].copy()
        print(modeDataset2['Crime type'].values[0])
        
        ## SEVERITY RATING MODEL
        dataset = dataset.loc[dataset['Outward Postcode'] == OutwardPostcodeVariable]
        dataset = dataset[['Longitude', 'Latitude', 'Severity', 'Month Number', 'Month in Unix', 'Year Number']].copy()
        X = dataset[['Month Number', 'Month in Unix', 'Year Number']]
        y =  dataset[['Severity']]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        y_pred = model.predict(X_test)
                               
        print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
        print('Root Mean squared error: %.2f' % np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))
        
        
        dataset2 = dataset2.loc[dataset2['Outward Postcode'] == OutwardPostcodeVariable2]
        dataset2 = dataset2[['Longitude', 'Latitude', 'Severity', 'Month Number', 'Month in Unix', 'Year Number']].copy()
        X2 = dataset2[['Month Number', 'Month in Unix', 'Year Number']]
        y2 =  dataset2[['Severity']]
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
        model2 = LinearRegression()
        model2.fit(X_train2, y_train2)
        model2.score(X_test2, y_test2)
        y_pred2 = model2.predict(X_test2)
                               
        print('Mean squared error: %.2f' % mean_squared_error(y_test2, y_pred2))
        print('Root Mean squared error: %.2f' % np.sqrt(mean_squared_error(y_test2, y_pred2)))
        print('Coefficient of determination: %.2f' % r2_score(y_test2, y_pred2))
        
        
        ##NUMBER OF CRIMES MODELLING
        NumberOfCrimes = dataset[['Month in Unix']].copy()
        NumberOfCrimes['Count'] = 0
        NumberOfCrimes['Count']=NumberOfCrimes.groupby(by='Month in Unix')['Month in Unix'].transform('count')
        NumberOfCrimes = NumberOfCrimes.drop_duplicates()
        XNumberofCrimes = NumberOfCrimes[['Month in Unix']]
        yNumberofCrimes =  NumberOfCrimes[['Count']]

        X_trainNumberofCrimes, X_testNumberofCrimes, y_trainNumberofCrimes, y_testNumberofCrimes = train_test_split(XNumberofCrimes, yNumberofCrimes, test_size=0.2, random_state=42)
        modelNumberofCrimes = LinearRegression()
        modelNumberofCrimes.fit(X_trainNumberofCrimes, y_trainNumberofCrimes)
        modelNumberofCrimes.score(X_testNumberofCrimes, y_testNumberofCrimes)
        y_predNumberofCrimes = modelNumberofCrimes.predict(X_testNumberofCrimes)

        print('Mean squared error: %.2f' % mean_squared_error(y_testNumberofCrimes, y_predNumberofCrimes))
        print('Root Mean squared error: %.2f' % np.sqrt(mean_squared_error(y_testNumberofCrimes, y_predNumberofCrimes)))
        print('Coefficient of determination: %.2f' % r2_score(y_testNumberofCrimes, y_predNumberofCrimes))

        NumberOfCrimes2 = dataset2[['Month in Unix']].copy()
        NumberOfCrimes2['Count'] = 0
        NumberOfCrimes2['Count']=NumberOfCrimes2.groupby(by='Month in Unix')['Month in Unix'].transform('count')
        NumberOfCrimes2 = NumberOfCrimes2.drop_duplicates()
        XNumberofCrimes2 = NumberOfCrimes2[['Month in Unix']]
        yNumberofCrimes2 =  NumberOfCrimes2[['Count']]

        X_trainNumberofCrimes2, X_testNumberofCrimes2, y_trainNumberofCrimes2, y_testNumberofCrimes2 = train_test_split(XNumberofCrimes2, yNumberofCrimes2, test_size=0.2, random_state=42)
        modelNumberofCrimes2 = LinearRegression()
        modelNumberofCrimes2.fit(X_trainNumberofCrimes2, y_trainNumberofCrimes2)
        modelNumberofCrimes2.score(X_testNumberofCrimes2, y_testNumberofCrimes2)
        y_predNumberofCrimes2 = modelNumberofCrimes2.predict(X_testNumberofCrimes2)

        print('Mean squared error: %.2f' % mean_squared_error(y_testNumberofCrimes2, y_predNumberofCrimes2))
        print('Root Mean squared error: %.2f' % np.sqrt(mean_squared_error(y_testNumberofCrimes2, y_predNumberofCrimes2)))
        print('Coefficient of determination: %.2f' % r2_score(y_testNumberofCrimes2, y_predNumberofCrimes2))



        
        NewMonthNumber = (datetime.utcfromtimestamp(NextMonthValue).strftime('%m'))
        NewYearNumber = (datetime.utcfromtimestamp(NextMonthValue).strftime('%Y'))
        
        NewMonthNumber2 = (datetime.utcfromtimestamp(NextMonthValue2).strftime('%m'))
        NewYearNumber2 = (datetime.utcfromtimestamp(NextMonthValue2).strftime('%Y'))
        

        severityRating = model.predict([[NewMonthNumber, NextMonthValue, NewYearNumber]])
        NumberOfCrimePrediction = modelNumberofCrimes.predict([[NextMonthValue]])

        print(severityRating[0][0])
        print(NumberOfCrimePrediction[0][0])
        
        severityRating2 = model2.predict([[NewMonthNumber2, NextMonthValue2, NewYearNumber2]])
        NumberOfCrimePrediction2 = modelNumberofCrimes2.predict([[NextMonthValue2]])
        
        if NumberOfCrimePrediction[0][0] < 0 :
            NumberOfCrimePrediction[0][0] = 0
            
        if NumberOfCrimePrediction2[0][0] < 0 :
            NumberOfCrimePrediction2[0][0] = 0
            
        print(severityRating2[0][0])
        print(NumberOfCrimePrediction2[0][0])
        
        message = {'latitudeHead':location1.latitude, 'longitudeHead':location1.longitude}
        
        print('-----Model-----')
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)
        print('-----ModelofCrimes-----')
        print('intercept:', modelNumberofCrimes.intercept_)
        print('slope:', modelNumberofCrimes.coef_)
        print(message)
        
        if modelNumberofCrimes.coef_ < 0:
            print('Decreasing')
            crimeIncreasingDecreasingVar = "Decreasing"
        else:
            print('Increasing')
            crimeIncreasingDecreasingVar = "Increasing"
            
        if modelNumberofCrimes2.coef_ < 0:
            print('Decreasing')
            crimeIncreasingDecreasingVar2 = "Decreasing"
        else:
            print('Increasing')
            crimeIncreasingDecreasingVar2 = "Increasing"
        #message = {'Longitude':location.longitude, "latitude":location.latitude}
        #message = jsonify(message)
        #message2 = jsonify(jsonPlottingValues)
        #message = message + message2
        PlottingDataset['SeverityRating'] = round(severityRating[0][0], 2)
        PlottingDataset['PredictedNumberOfCrimes'] = round(NumberOfCrimePrediction[0][0], 0)
        PlottingDataset['ModeCrime'] = modeDataset['Crime type'].values[0]
        PlottingDataset['inputPostcode'] = postcodeVariable
        PlottingDataset['inputPostcodeOutward'] = OutwardPostcodeVariable

        PlottingDataset['SeverityRating2'] = round(severityRating2[0][0], 2)
        PlottingDataset['PredictedNumberOfCrimes2'] = round(NumberOfCrimePrediction2[0][0], 0)
        PlottingDataset['ModeCrime2'] = modeDataset2['Crime type'].values[0]
        PlottingDataset['inputPostcode2'] = postcodeVariable2
        PlottingDataset['inputPostcodeOutward2'] = OutwardPostcodeVariable2
        
        
        PlottingDataset['NumberInwardCrimes'] = NumberInwardCrimes
        PlottingDataset['NumberInwardCrimes2'] = NumberInwardCrimes2
        
        PlottingDataset['ModeCrimeInward'] = modeDatasetInward['Crime type'].values[0]
        PlottingDataset['ModeCrimeInward2'] = modeDatasetInward2['Crime type'].values[0]
        
        
        PlottingDataset['NumberOutwardCrimes'] = NumberInwardCrimesOutward
        PlottingDataset['NumberOutwardCrimes2'] = NumberInwardCrimesOutward2
        
        PlottingDataset['FirstMonth'] = SmallestDate
        PlottingDataset['FirstYear'] = LargestDate
        
        PlottingDataset['SecondMonth'] = SmallestDate2
        PlottingDataset['SecondYear'] = LargestDate2
        
        PlottingDataset['CrimeIncreasingDecreasing'] = crimeIncreasingDecreasingVar
        PlottingDataset['CrimeIncreasingDecreasing2'] = crimeIncreasingDecreasingVar2
        
        datasetReturned = PlottingDataset[['SeverityRating', 'PredictedNumberOfCrimes', 'ModeCrime', 'inputPostcode', 'inputPostcodeOutward', 'SeverityRating2', 'PredictedNumberOfCrimes2', 'ModeCrime2', 'inputPostcode2', 'inputPostcodeOutward2', 'NumberInwardCrimes', 'NumberInwardCrimes2', 'ModeCrimeInward', 'ModeCrimeInward2', 'NumberOutwardCrimes', 'NumberOutwardCrimes2', 'FirstMonth', 'FirstYear', 'SecondMonth', 'SecondYear', 'CrimeIncreasingDecreasing', 'CrimeIncreasingDecreasing2']].copy()
    
        datasetReturned = datasetReturned.iloc[:1]
        print(datasetReturned)
        jsonPlottingValues = datasetReturned.to_json(orient="records")
        return jsonPlottingValues  # serialize and use JSON headers      

if __name__ == "__main__":
    application.run(debug=True)