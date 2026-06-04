import argparse
import os

import pandas as pd


CITY_BBOXES = {
    # min_lat, max_lat, min_lng, max_lng
    "austin": (30.10, 30.45, -97.95, -97.55),
    "nyc": (40.49, 40.92, -74.27, -73.68),
    "sf_bay": (37.20, 38.10, -122.70, -121.70),
    "seattle": (47.30, 47.80, -122.50, -122.10),
    "la": (33.70, 34.40, -118.70, -117.90),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to SNAP loc-gowalla_totalCheckins.txt.gz or an uncompressed TSV file.",
    )
    parser.add_argument("--output", default="data/checkins-gowalla-austin.txt")
    parser.add_argument("--city", default="austin", choices=sorted(CITY_BBOXES))
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LAT", "MAX_LAT", "MIN_LNG", "MAX_LNG"),
        help="Override city bbox.",
    )
    parser.add_argument("--min-checkins", default=101, type=int)
    parser.add_argument("--max-users", default=0, type=int, help="Keep top-N active users after filtering; 0 keeps all.")
    parser.add_argument("--chunksize", default=500_000, type=int)
    return parser.parse_args()


def read_city_chunks(path, bbox, chunksize):
    min_lat, max_lat, min_lng, max_lng = bbox
    names = ["userid", "datetime", "lat", "lng", "placeid"]
    chunks = []
    reader = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=names,
        compression="infer",
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk.dropna(subset=names)
        chunk = chunk[
            chunk["lat"].between(min_lat, max_lat)
            & chunk["lng"].between(min_lng, max_lng)
        ]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(columns=names)
    return pd.concat(chunks, ignore_index=True)


def main():
    args = parse_args()
    bbox = tuple(args.bbox) if args.bbox else CITY_BBOXES[args.city]

    df = read_city_chunks(args.input, bbox, args.chunksize)
    if df.empty:
        raise SystemExit(f"No check-ins found for city={args.city} bbox={bbox}")

    df["userid"] = pd.to_numeric(df["userid"], errors="coerce").astype("Int64")
    df["placeid"] = pd.to_numeric(df["placeid"], errors="coerce").astype("Int64")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["userid", "placeid", "lat", "lng", "datetime"]).drop_duplicates()
    df["userid"] = df["userid"].astype(int)
    df["placeid"] = df["placeid"].astype(int)

    user_counts = df.groupby("userid").size().sort_values(ascending=False)
    active_users = user_counts[user_counts >= args.min_checkins]
    if args.max_users > 0:
        active_users = active_users.head(args.max_users)
    df = df[df["userid"].isin(active_users.index)].copy()
    if df.empty:
        raise SystemExit(
            f"No users left after min_checkins={args.min_checkins}. "
            "Try a larger bbox/city or a smaller threshold."
        )

    # Flashback loader expects check-ins for each user to be contiguous and in
    # descending chronological order; it reverses each user sequence internally.
    df = df.sort_values(["userid", "datetime"], ascending=[True, False])
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df[["userid", "datetime", "lat", "lng", "placeid"]].to_csv(
        args.output,
        sep="\t",
        header=False,
        index=False,
    )

    print("City:", args.city)
    print("BBox:", bbox)
    print("Output:", args.output)
    print("Check-ins:", len(df))
    print("Users:", df["userid"].nunique())
    print("POIs:", df["placeid"].nunique())
    print("Date min:", df["datetime"].min())
    print("Date max:", df["datetime"].max())


if __name__ == "__main__":
    main()
