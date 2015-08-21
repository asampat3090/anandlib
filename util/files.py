import os

def create_path(path):
    """
    Creates the directory if not present or
    if filepath given, creates the directory of file
    """
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def cleanup_string(s):
    """
    Cleanup string formatting
    """
    if s is None:
        return ''
    else:
        # Can't figure out a general solution....just error out for now
        try:
            result = s.decode('ascii', 'ignore')
            return result
        except:
            return '...error decoding...'