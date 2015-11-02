from datetime import datetime
from dateutil import tz

def convert_utc_str_to_local_dt(utc_string):
    """
    Reads UTC string and outputs datetime object in local time zone.
    """
    if utc_string[-1] == 'Z':
        utc_string = utc_string[:-1]

    from_zone = tz.tzutc()
    to_zone = tz.tzlocal()

    utc = datetime.strptime(utc_string, '%Y-%m-%dT%H:%M:%S.%f')

    # Tell the datetime object that it's in UTC time zone since
    # datetime objects are 'naive' by default
    utc = utc.replace(tzinfo=from_zone)

    # Convert time zone
    new_datetime = utc.astimezone(to_zone)
    return new_datetime