import csv

with open("../../data/ad.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)
