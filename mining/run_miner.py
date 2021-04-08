from subprocess import Popen
import time
import os

import sys
# from keyboard import press, press_and_release
from pynput.keyboard import Key, Controller
# from pyautogui import press

miner = sys.argv[1]

# ethereum
if miner.lower() == 'eth':

    miner_path = '/Users/Matt/scripts/mining/lolMiner_v1.21_Win64/1.21/'
    miner_file = 'mine_eth.bat'

# xmr
elif miner.lower() == 'xmr':

    # xmrig
    miner_path = '/Users/Matt/scripts/mining/xmrig-6.9.0-gcc-win64/xmrig-6.9.0/'
    miner_file = 'xmrig.exe'

    # miner_path = '/Users/Matt/Scripts/mining/nanominer-windows-3.2.2/'
    # miner_file = 'nanominer.exe'


os.chdir(miner_path)



keyboard = Controller()

while True:

    print('\n****** Mining ' + miner.upper() + ' ******\n')

    proc = Popen(miner_path + miner_file)

    time.sleep(360*60) # run for 6 hours

    proc.kill() # kill the process

    break # break out of the while loop
    print('\n****** Process Killed - restarting ******\n')
    time.sleep(3)

stdout, stderr = proc.communicate()
