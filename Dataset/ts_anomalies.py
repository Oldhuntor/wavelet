from gutenTAG import GutenTAG, TrainingType, LABEL_COLUMN_NAME


config = {
    "timeseries": [
        {
            "name": "test",
            "length": 100,
            "base-oscillations": [
                {"kind": "sine"}
            ],
            "anomalies": [
                {"length": 5, "channel": 0, "kinds": [{"kind": "mean", "offset": .5}]}
            ]
        }
    ]
}
gutentag = GutenTAG(seed=1)
gutentag.load_config_dict(config)

# call generate() to create the datasets (in-memory)
datasets = gutentag.generate(return_timeseries=True)

# we only defined a single test time series
assert len(datasets) == 1
d = datasets[0]
assert d.name == "test"
assert d.training_type == TrainingType.TEST

# the data points are stored at
df = d.timeseries
df.iloc[:, 1:-1]
# the labels are stored at
df[LABEL_COLUMN_NAME]

print(df.head())