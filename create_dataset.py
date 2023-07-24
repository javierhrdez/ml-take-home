import pandas as pd
from pymongo import MongoClient
import json
from zipfile import ZipFile

conn = MongoClient()
db = conn.ml

X_train_items = db.X_train
y_train_items = db.y_train
X_test_items = db.X_test
y_test_items = db.y_test


with ZipFile("MLA_100k.jsonlines.zip", 'r') as zip:
    # printing all the contents of the zip file
    zip.printdir()
    # extracting all the files in current working directory 
    zip.extractall(path = ".")

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    data = [json.loads(x) for x in open("MLA_100k.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    return X_train, y_train, X_test, y_test

X_train_raw, y_train_raw, X_test_raw, y_test_raw  = build_dataset()

X_train_items.insert_many(X_train_raw)
X_test_items.insert_many(X_test_raw)

X_train_collect = X_train_items.find()
X_test_collect = X_test_items.find()

X_train = pd.DataFrame(X_train_collect)
X_test = pd.DataFrame(X_test_collect)

X_train.pop("_id")
X_test.pop("_id")

y_train = pd.DataFrame({"is_new":y_train_raw})
y_test = pd.DataFrame({"is_new":y_test_raw})

df_train = pd.concat([X_train,y_train], axis = 1)
df_test = pd.concat([X_test,y_test], axis = 1)

df_train.to_csv("dataset/df_train.csv.gzip" , index=False, compression="gzip" )
df_test.to_csv("dataset/df_test.csv.gzip",  index=False, compression="gzip")