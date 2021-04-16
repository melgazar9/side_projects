from subprocess import Popen, call
import time
import os

import sys
# from keyboard import press, press_and_release
from pynput.keyboard import Key, Controller
# from pyautogui import press

miner = sys.argv[1]

# ethereum
if miner.lower() == 'eth':

    #miner_path = '/Users/Matt/scripts/mining/lolMiner_v1.21_Win64/1.21/' # local windows machine
    #miner_file = 'mine_eth.bat' # windows
    miner_path = '/home/melgazar9/Downloads/lolMiner_v1.25_Lin64/1.25/' # local linux machine
    miner_file = 'mine_eth.sh'

# xmr
elif miner.lower() == 'xmr':

    # xmrig
    #miner_path = '/Users/Matt/scripts/mining/xmrig-6.9.0-gcc-win64/xmrig-6.9.0/' # local windows machine
    miner_path = '/home/melgazar9/Downloads/xmrig/build/' # local linux machine
    #miner_file = 'xmrig.exe' # windows
    miner_file = 'xmrig'

    # miner_path = '/Users/Matt/Scripts/mining/nanominer-windows-3.2.2/'
    # miner_file = 'nanominer.exe'
#print(miner_path + miner_file)

os.chdir(miner_path)
keyboard = Controller()

while True:

    print('\n****** Mining ' + miner.upper() + ' ******\n')

    if miner.lower() == 'xmr':
        proc = call(['sudo', miner_path + miner_file])
    else:
        proc = Popen(miner_path + miner_file)

    time.sleep(360*60) # run for 6 hours

    proc.kill() # kill the process

    break # break out of the while loop
    print('\n****** Process Killed - restarting ******\n')
    time.sleep(3)

stdout, stderr = proc.communicate()
