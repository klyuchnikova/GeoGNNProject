from .dataset import SequenceSample

class SequenceBuilder:
    def __init__(self, seq_len=20, stride=1):
        self.seq_len = seq_len
        self.stride = stride

    def build(self, df):
        samples = []
        df = df.sort_values("timestamp")

        for uid, user_df in (df.groupby("user_id")):
            pois = user_df.poi_id.tolist()
            cats = user_df.category.tolist()
            ts = user_df.timestamp.tolist()
            lats = user_df.lat.tolist()
            lons = user_df.lon.tolist()
            for start in range(0, len(pois)-self.seq_len, self.stride):
                end = start + self.seq_len
                samples.append(
                    SequenceSample(
                        user_id=uid,
                        poi_seq=pois[start:end],
                        category_seq=cats[start:end],
                        timestamp_seq=ts[start:end],
                        lat_seq=lats[start:end],
                        lon_seq=lons[start:end],
                        target_poi=pois[end],
                    )
                )

        return samples
