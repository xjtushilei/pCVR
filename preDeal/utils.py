import datetime
import scipy as sp


def get_how_much_time(time_str, year_month='2017-01', start_date_time='2017-01-010000'):
    """
    通过输入xxxxxx格式的时间，得到一个时间差。单位是秒
    """
    t_str = year_month + "-" + time_str
    t1 = datetime.datetime.strptime(t_str, '%Y-%m-%d%H%M')
    t2 = datetime.datetime.strptime(start_date_time, '%Y-%m-%d%H%M')
    how_long = t1.timestamp() - t2.timestamp()
    return how_long


def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll
