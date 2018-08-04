import datetime

# print("DEBBUG: date_time = " + str(datetime.datetime.now()).replace(" ","_"))

import hashlib
s = 'Y Minh Le'
# int(hashlib.sha1(s).hexdigest(), 16) % (10 ** 8)
id = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % (10**8)
print("id = {}".format(id))