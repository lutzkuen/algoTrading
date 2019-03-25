from observer import limitcheck
import time
while True:
    limitcheck.checkaccount('/home/tubuntu/settings_triangle.conf', 'demo')
    limitcheck.checkaccount('/home/tubuntu/settings_triangle.conf', 'live')
    time.sleep(60)
