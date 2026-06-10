import argparse
from flashback.data.cities import rank_cities
p=argparse.ArgumentParser();p.add_argument("--checkins",default="data/raw/loc-gowalla_totalCheckins.txt.gz");p.add_argument("--min-checkins",type=int,default=101)
a=p.parse_args();print(rank_cities("gowalla",a.checkins,["new_york","los_angeles","chicago","san_francisco","austin","dallas","seattle","boston"],a.min_checkins).to_string(index=False))
