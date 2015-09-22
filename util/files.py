import os
import sys

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

# http://stackoverflow.com/a/13895723/2626090
def reporthook(blocknum, blocksize, totalsize):
    """
    Progress bar callback function for urlretrieve
    """
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))