import subprocess
from utils import *
from get_instances import *
from ruamel.yaml import YAML

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

def run_epochs_varnet(epochs, yaml_file_path, shell_file_path, shell_file_train, shell_file_test, k):
    
    if k == 1:
        modify_shell_script(shell_file_train, "configs/base_varnet,k=1.yaml")
        modify_shell_script(shell_file_test, "configs/base_varnet,k=1.yaml")
    else:
        modify_shell_script(shell_file_train, "configs/base_varnet,k=10.yaml")
        modify_shell_script(shell_file_test, "configs/base_varnet,k=10.yaml")
        
    for epoch in range(epochs):
        
        if epoch == 0 and k == 1:
            modify_yaml(yaml_file_path, 'restore_weights', False)
        elif epoch == 0 and k == 10:
            modify_yaml(yaml_file_path, 'restore_path', 'workspace/base_varnet,k=1/checkpoints/')        
        elif epoch != 0 and k == 10:
            modify_yaml(yaml_file_path, 'restore_path', 'workspace/base_varnet,k=10/checkpoints/')        
        else:
            modify_yaml(yaml_file_path, 'restore_weights', 'final')
            
        # # Number of times to run the shell script
        subprocess.run(shell_file_path)    

if __name__ == "__main__":
    # Path to your shell script
    shell_varnet = './scripts/batch_varnet.sh'       
    shell_varnet_train = './scripts/train_varnet.sh'
    shell_varnet_test = './scripts/test_varnet.sh'
    yaml_varnet_k_1 = './configs/base_varnet,k=1.yaml'
    yaml_varnet_k_10 = './configs/base_varnet,k=10.yaml'

    # the epochs number defines how many (train.py + test.py) iterations
    epochs = 1    
    
    ## run varnet ##  k = 1
    # yaml_file = yaml_varnet_k_1
    # run_epochs_varnet(epochs, yaml_file, shell_varnet, shell_varnet_train, shell_varnet_test, k=1)
    
    ## run varnet ##  k = 10
    yaml_file = yaml_varnet_k_10
    run_epochs_varnet(epochs, yaml_file, shell_varnet, shell_varnet_train, shell_varnet_test, k=10)
    

