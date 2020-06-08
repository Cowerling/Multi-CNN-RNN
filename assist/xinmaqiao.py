import os
import csv

data_dir = '/opt/data'
file_name = os.path.join(data_dir, 'ground_water/xinmaqiao.csv')

data = {}

with open(file_name, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)

    count = 0

    for row in reader:
        date = row[0].split(' ')[0]
        ground_water_level = float(row[2])

        if date not in data.keys():
            data[date] = ground_water_level
            count = 0
        else:
            data[date] += ground_water_level

        count += 1
        data[date] /= count

for (date, ground_water_level) in data.items():
    print(date, ground_water_level)
