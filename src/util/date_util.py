import time
from datetime import datetime


def get_now(): return datetime.now()

'''
get_time_str("%Y%m%d")
get_time_str("%Y%m%d%H%M%S")
'''
def get_time_str(FORMAT_DATETIME="%Y%m%d%H%M%S"):
    return datetime.fromtimestamp(int(time.time())).strftime(FORMAT_DATETIME)

def get_time_str16(FORMAT_DATETIME="%Y-%m-%d %H:%M"):
    return datetime.fromtimestamp(int(time.time())).strftime(FORMAT_DATETIME)

def get_time_str15(FORMAT_DATETIME="%Y-%m-%d_%H%M"):
    return datetime.fromtimestamp(int(time.time())).strftime(FORMAT_DATETIME)

def get_time():
    return datetime.fromtimestamp(int(time.time()))

'''
now = datetime.now()
past = datetime.strptime("20210305", "%Y%m%d")
get_diff_time_sec(past, now)
'''

def get_diff_time_microseconds(start_tm, end_tm):
    return (end_tm - start_tm).seconds + ((end_tm - start_tm).microseconds/1000000)

def get_diff_time_sec(start_tm, end_tm):
    return (end_tm - start_tm).seconds

def get_diff_time_min(start_tm, end_tm):
    return int((end_tm - start_tm).seconds / 60)

def get_diff_time_hour(start_tm, end_tm):
    return int((end_tm - start_tm).seconds / 3600)

def get_diff_time_day(start_tm, end_tm):
    return (end_tm - start_tm).days


# print(get_time_str("%Y%m%d"))
# print(get_time())
# now = datetime.now()
# past = datetime.strptime("2021-03-20 21:09:00", "%Y-%m-%d %H:%M:%S")
# print("now=",now)
# print("past=",past)
# print("get_diff_time_sec=",get_diff_time_sec(past, now))
# print("get_diff_time_microseconds=",get_diff_time_microseconds(past, now))
# print("get_diff_time_microseconds=",get_diff_time_sec(past, now) + get_diff_time_microseconds(past, now)/1000000)
