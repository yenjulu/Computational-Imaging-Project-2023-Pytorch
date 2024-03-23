import subprocess
from utils import *
from get_instances import *
from ruamel.yaml import YAML
# import sys
# sys.path.append(os.path.abspath('../'))

yaml = YAML()
yaml.preserve_quotes = True  # Preserve the original quotes
yaml.indent(mapping=2, sequence=4, offset=2)  # Set indentation preferences

def modify_yaml(file_path, key, new_value):
    with open(file_path, 'r') as file:
        data = yaml.load(file)

    data[key] = new_value  # Modify only the specific key

    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def modify_shell_script(shell_script_path, new_value):
    with open(shell_script_path, 'r') as file:
        lines = file.readlines()

    with open(shell_script_path, 'w') as file:
        for line in lines:
            if line.startswith('TRAIN_CONFIG_YAML='):
                file.write(f'TRAIN_CONFIG_YAML="{new_value}"\n')
            elif line.startswith('TEST_CONFIG_YAML='):
                file.write(f'TEST_CONFIG_YAML="{new_value}"\n')
            else:
                file.write(line)

def run_file(yaml_file_path, shell_file_path, k, n):
        
    modify_yaml(yaml_file_path, 'config_name', f'fastmri_modl,k={k},n={n}')
    modify_yaml(yaml_file_path, 'description', f'"fastmri config, k={k},n={n},sigma:0.01"')
    modify_yaml(yaml_file_path, 'k_iters', k)  
    modify_yaml(yaml_file_path, 'n_layers', n) 
    modify_yaml(yaml_file_path, 'restore_weights', 'best') 
    modify_yaml(yaml_file_path, 'restore_path', f'workspace/fastmri_modl,k=1,n={n}/checkpoints/')  
    subprocess.run(shell_file_path)

def run_file_0(yaml_file_path, shell_file_path, k, n):
        
    modify_yaml(yaml_file_path, 'config_name', f'fastmri_modl,k={k},n={n}')
    modify_yaml(yaml_file_path, 'description', f'"fastmri config, k={k},n={n},sigma:0.01"')
    modify_yaml(yaml_file_path, 'k_iters', k)  
    modify_yaml(yaml_file_path, 'n_layers', n) 
    modify_yaml(yaml_file_path, 'restore_weights', False)  
    # modify_yaml(yaml_file_path, 'restore_weights', 'best') 
    modify_yaml(yaml_file_path, 'restore_path', f'workspace/fastmri_modl,k={k},n={n}/checkpoints/')  
    subprocess.run(shell_file_path)
    
    
if __name__ == "__main__":
    # Path to your shell script
    shell_modl = '/home/woody/rzku/mlvl125h/MoDL_PyTorch/scripts/test.sh' 
    #shell_modl = '/home/woody/rzku/mlvl125h/MoDL_PyTorch/scripts/train.sh'      
    yaml_modl = '/home/woody/rzku/mlvl125h/MoDL_PyTorch/configs/fastmri_modl.yaml'
        
    
    ## run modl ## 
    yaml_file = yaml_modl
    n = 5
    for k in range(2,12,1):
        if k == 1:
            run_file_0(yaml_file, shell_modl, k, n)  
        
        else:
            run_file(yaml_file, shell_modl, k, n)

    # n = 5
    # for k in range(7,13,1):
    #     run_file(yaml_file, shell_modl, k, n)