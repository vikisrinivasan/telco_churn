import datetime
import calendar

def diff_month(start_date,end_date):
    return (start_date.year-end_date.year)*12+start_date.month-end_date.month

def last_day_of_month(yyyymm):
    day=datetime.datetime.strptime(str(yyyymm),"%Y-%m-%d")
    day=day.replace(day=calendar.monthrange(day.year,day.month)[1])
    return day