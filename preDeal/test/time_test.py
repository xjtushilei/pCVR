import datetime

t_str ="2017-01"+"-"+ '170000'
d = datetime.datetime.strptime(t_str, '%Y-%m-%d%H%M')
print(d.timestamp())