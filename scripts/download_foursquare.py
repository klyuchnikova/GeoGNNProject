import argparse
from flashback.data.downloads import download_foursquare_kaggle
p=argparse.ArgumentParser();p.add_argument("--dataset",default="chetanism/foursquare-nyc-and-tokyo-checkin-dataset");p.add_argument("--output",default="data/raw/foursquare")
a=p.parse_args();print(download_foursquare_kaggle(a.dataset,a.output))
