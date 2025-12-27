import json
from datetime import datetime

# Check 24h timeseries
with open('mongo_data/timeseries/2_24h.json') as f:
    data_24h = json.load(f)

print(f'24h timeseries records: {len(data_24h)}')
first = datetime.fromtimestamp(data_24h[0]['timestamp'])
last = datetime.fromtimestamp(data_24h[-1]['timestamp'])
days = (data_24h[-1]['timestamp'] - data_24h[0]['timestamp']) / 86400
print(f'Date range: {first.strftime("%Y-%m-%d")} to {last.strftime("%Y-%m-%d")}')
print(f'Days of data: {days:.0f}')

# Check 1h timeseries
with open('mongo_data/timeseries/2_1h.json') as f:
    data_1h = json.load(f)

print(f'\n1h timeseries records: {len(data_1h)}')
first_1h = datetime.fromtimestamp(data_1h[0]['timestamp'])
last_1h = datetime.fromtimestamp(data_1h[-1]['timestamp'])
days_1h = (data_1h[-1]['timestamp'] - data_1h[0]['timestamp']) / 86400
print(f'Date range: {first_1h.strftime("%Y-%m-%d")} to {last_1h.strftime("%Y-%m-%d")}')
print(f'Days of data: {days_1h:.0f}')

# Check 5m snapshot
with open('mongo_data/prices_5m.json') as f:
    prices_5m = json.load(f)

print(f'\n5m snapshot: {len(prices_5m)} records (current snapshot only)')
