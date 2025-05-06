cu_list = [118, 121, 124, 126, 128]
cp_list = [39, 310, 311, 312, 313]
torch_list = ["2.0.1", "2.1.2", "2.2.2", "2.3.1", "2.4.1", "2.5.1", "2.6.0", "2.7.0"]

def is_valid_combo(cu, torch, cp):
    if (torch == "2.0.1" and cu == 118 and cp < 312) or \
            (torch == "2.1.2" and cu < 124 and cp < 312) or \
            (torch == "2.2.2" and cu < 124 and cp < 313) or \
            (torch == "2.3.1" and cu < 124 and cp < 313) or \
            (torch == "2.4.1" and cu < 124 and cp < 313) or \
            (torch == "2.5.1" and cu < 126) or \
            (torch == "2.6.0" and cu >= 126) or \
            (torch == "2.7.0" and cu >= 126):
        print(f'- cuda: {cu}')
        print(f'  torch: {torch}')
        print(f'  python: {cp}')
        return True
    return False

task_map = {}

for cu in reversed(cu_list):
    task_map[str(cu)] = {}
    for cp in reversed(cp_list):
        task_map[str(cu)][str(cp)] = {}
        for torch in reversed(torch_list):
            task_map[str(cu)][str(cp)][torch] = int(is_valid_combo(cu, torch, cp))

