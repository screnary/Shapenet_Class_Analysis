import os
from datetime import datetime

def print_now():
    # With format 2020-11-26 10:25:09
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_process(i, total):
    return '| {:<5} : {}'.format(i, total)


DATA_PARAMS = ['02691156', '02958343']
BASE_CMD = 'python shapenet_prepare.py --mode {} --synoffset {}'

total_len = len(DATA_PARAMS)
g_step = 0

for param in DATA_PARAMS:
    cmd = BASE_CMD.format('sampling', param)
    print(print_now(), '='*30, print_process(g_step, total_len))
    print(cmd)
    print('='*60)
    os.system(cmd)
    g_step += 1

print('\n\n', print_now(), '\n All done.')

