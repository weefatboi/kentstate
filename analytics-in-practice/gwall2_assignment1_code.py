import pandas as pd
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 30)

## Exercise 1
agency = pd.read_csv('C:/Users/Owner/Documents/Datasets/us_agencies.csv')
company = pd.read_csv('C:/Users/Owner/Documents/Datasets/us_companies.csv')

## Column Exploration
print(agency.columns)
print(agency.shape)
print(company.columns)
print(company.shape)

agency_missing = agency.isna().mean().round(4) * 100
print(agency_missing)

company_missing = company.isna().mean().round(4) * 100
print(company_missing)

## Exercise 2

import json
import ijson
filename = "C:/Users/Owner/Documents/Datasets/ChicagoTraffic.json"
with open(filename, 'r') as x:
    objects = ijson.items(x, 'meta.view.columns.item')
    columns = list(objects)

column_names = [col["fieldName"] for col in columns]
print(*column_names, sep="\n")

good_columns = [
    "id",
    "traffic_volume_count_location_address",
    "street",
    "date_of_count",
    "total_passing_vehicle_volume",
    "vehicle_volume_by_each_direction_of_traffic",
    "latitude",
    "longitude",
    "location"]
data = []
with open(filename, 'r') as x:
    objects = ijson.items(x, 'data.item')
    for row in objects:
        selected_row = []
        for item in good_columns:
            selected_row.append(row[column_names.index(item)])
            data.append(selected_row)
print(*good_columns, sep="\n")

traffic = pd.DataFrame(data, columns=good_columns)

traffic['total_passing_vehicle_volume'] = pd.to_numeric(traffic['total_passing_vehicle_volume'])
unique_streets = traffic.groupby(traffic['street']).total_passing_vehicle_volume.sum()
print(unique_streets[:'115th St'])
print(unique_streets[:'115th St'].sum())

geolocation1 = traffic[(traffic['latitude'] == '41.651861') & (traffic['longitude'] == '-87.54501')].total_passing_vehicle_volume.sum()
geolocation2 = traffic[(traffic['latitude'] == '41.66836') & (traffic['longitude'] == '-87.620176')].total_passing_vehicle_volume.sum()
print(geolocation1)
print(geolocation2)