import datetime


LOG_LEVELS = {
    0: 'DEBUG',
    10: 'INFO',
    20: 'MODEL',
    30: '\U000026A0 WARNING',
    40: '\U0000274C ERROR',
    50: '\U0000274C CRITICAL'
}


def timestamp(log_level_value=10):
    assert log_level_value in [num for num in LOG_LEVELS.keys()]
    log_level = LOG_LEVELS.get(log_level_value, 'Invalid log level')
    log_level = "[" + log_level + "]"
    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S") + " " + log_level + " "
    return timestamp_str


if __name__ == '__main__':
    print(f'{timestamp(30)}This is a warning!')
    print(f'{timestamp(40)}This is an error!')