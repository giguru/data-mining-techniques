from utils import get_temporal_records, read_data, SECONDS_IN_DAY


data = read_data()


records = get_temporal_records(data, SECONDS_IN_DAY * 5, {
# Just naively take the average of all data points
    'circumplex.arousal': 'mean',
    'circumplex.valence': 'mean',
    'activity': 'mean',
    'screen': 'mean',
    'call': 'sum',
    'sms': 'sum',
})