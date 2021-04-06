import os
import subprocess
import psutil

action_list = ['move']
fluent_list = ['at', 'cost', 'picked']


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def extract_result(result_file):
    with open(result_file) as infile:
        content = infile.readlines()
        for info in content:
            if info.startswith('Answer'):
                result = content[content.index(info) + 1].strip('\n')
                return result.split(" ")
    return None


def get_type(input_string):
    prefix = input_string[:input_string.find('(')]
    for act in action_list:
        if act == prefix:
            return "action"
    for flu in fluent_list:
        if flu == prefix:
            return "fluent"


def split_time(result):
    splitted_tuple = []
    for res in result:
        if "=" in res:
            equalposition = res.rfind('=')
            value = res[equalposition + 1:]
            timestamp = res[5:equalposition - 1]
            atom = "cost=" + value
            splitted_tuple.append((int(timestamp), atom, get_type(res)))
        else:
            index = res.rfind(',')
            timestamp = res[index + 1:][:-1]
            atompart = res[:index]
            atom = "".join(atompart) + ")"
            splitted_tuple.append((int(timestamp), atom, get_type(res)))
    return splitted_tuple


def construct_lists(split, step):
    actions = ''
    fluents = ''
    for s in split:
        if s[0] == step:
            if s[2] == 'action':
                actions = actions + s[1] + ' '
            else:
                fluents = fluents + s[1] + ' '
    return actions, fluents


def compute_plan(clingo_path, initial, goal, planning, q_value, constraint, log_dir, verbose=False):
    assert clingo_path is not None, 'clingo path must be specified'

    if verbose:
        print("[INFO] Generating symbolic plan...")

    # create the directory for planning
    log_dir = os.path.join(log_dir, 'planning_tmp')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    show = os.path.join(log_dir, 'show.lp')
    tmp_result_path = os.path.join(log_dir, 'result.tmp')
    files = initial + " " + planning + " " + q_value + " " + constraint + " " + goal + " " + show

    print('[INFO] Executing clingo command: ', clingo_path + " " + files + " --time-limit=180 > " + tmp_result_path)
    clingo_process = subprocess.Popen(clingo_path + " " + files + " --time-limit=180 > " + tmp_result_path,
                                      shell=True)
    p = psutil.Process(clingo_process.pid)
    try:
        p.wait(timeout=360)
    except psutil.TimeoutExpired:
        p.kill()
        print(bcolors.FAIL + "Planning timeout. Process killed." + bcolors.ENDC)
        return None

    result = extract_result(tmp_result_path)
    if result is None:
        return None
    split = split_time(result)

    max_time = int(sorted(split, key=lambda item: item[0], reverse=True)[0][0])
    if verbose:
        print("[INFO] Find a plan in", max_time, "steps")

    plan_trace = []
    for i in range(1, max_time + 1):
        actions, fluents = construct_lists(split, i)
        plan_trace.append((i, actions, fluents))
        if verbose is True:
            print(bcolors.OKBLUE + "[TIME STAMP]", i, bcolors.ENDC)
            if fluents != '':
                print(bcolors.OKGREEN + "[FLUENTS]" + bcolors.ENDC, fluents)
            if actions != '':
                print(bcolors.OKGREEN + "[ACTIONS]" + bcolors.ENDC, bcolors.BOLD + actions + bcolors.ENDC)
    return plan_trace
