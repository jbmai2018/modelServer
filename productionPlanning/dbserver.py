import pymongo
import datetime
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = "productionPlanning/uploads"
## Create MongoDB Client
myclient = pymongo.MongoClient("mongodb://localhost:27017")
mydb = myclient["jbmDB"]
mycoll = mydb["ppGeneration"]

def store_in_db(response):
    print("Storing in DB")
    ## Query if pKey already exists
    myquery = {"pKey": datetime.datetime.now().date().strftime('%d-%m-%y')}
    mydoc = mycoll.find(myquery)


    if not [x for x in mydoc]:
        print("Reading Excel")
        df = pd.read_excel(os.path.join(BASE_DIR, data_dir + response['file']))
        mydict = {"pKey": datetime.datetime.now().date().strftime('%d-%m-%y'),
                  "critical_sheet_j1": df.to_json(orient="records") if response['remark']=="J1" else {},
                  "critical_sheet_j3": df.to_json(orient="records") if response['remark']=="J3" else {},
                  "predicted_plan": {},
                  "plantID": {},
                  "lineID": "ISGEC-4"
                  }
        x = mycoll.insert_one(mydict)
    else:
        print("updating Excel")
        update_file(response)



def update_file(response):
    df = pd.read_excel(os.path.join(BASE_DIR, data_dir + response['file']))
    myquery = {"pKey": datetime.datetime.now().date().strftime('%d-%m-%y')}
    if response['remark']=="J1":
        newvalues = {"$set": {"critical_sheet_j1":df.to_json(orient='records') }}
    elif response['remark']=="J3":
        newvalues = {"$set": {"critical_sheet_j3": df.to_json(orient='records')}}
    mycoll.update_one(myquery, newvalues)

def get_data(key):
    myquery = {"pKey": datetime.datetime.now().date().strftime('%d-%m-%y')}
    mydoc = mycoll.find(myquery)
    return [x for x in mydoc][0][key]


def update_predicted_plan(predicted_plan):
    myquery = {"pKey": datetime.datetime.now().date().strftime('%d-%m-%y')}
    newvalues = {"$set": {"predicted_plan": predicted_plan}}
    mycoll.update_one(myquery,newvalues)