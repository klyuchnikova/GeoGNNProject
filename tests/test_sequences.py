import pandas as pd
from flashback.data.sequences import NextPoiSequenceDataset

def frame():
    n=12
    return pd.DataFrame({"user_id":[0]*n,"poi_id":list(range(n)),"timestamp":pd.date_range("2020-01-01",periods=n,freq="h",tz="UTC"),
        "latitude":[1.]*n,"longitude":[2.]*n,"split":["train"]*7+["validation"]*2+["test"]*3})

def test_validation_can_use_train_context():
    ds=NextPoiSequenceDataset(frame(),"validation",4,4)
    assert sum(int(x["target_mask"].sum()) for x in ds)==2

def test_targets_not_duplicated():
    ds=NextPoiSequenceDataset(frame(),"test",4,2)
    assert sum(int(x["target_mask"].sum()) for x in ds)==3
