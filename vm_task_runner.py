from datetime import datetime
import json
import subprocess
import time
import sys
import logging

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

SOURCE_VM_NAME = "compute-vm-28-119-150-ssd-1774434752419"
SOURCE_VM_USER = "helen-markova"
SOURCE_SCRIPT_PATH = "/home/helen-markova/ask_me/data_handler.py"
SSH_KEY_PATH = "/home/helen-markova/.ssh/id_rsa"  
LOG_FILE = "/home/helen-markova/vm_automation.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def run_command(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logging.error(f"Command timed out: {cmd}")
        return False, "", "Timeout"

def get_vm_status():
    success, stdout, _ = run_command(f"yc compute instance get {SOURCE_VM_NAME} --format json")
    info = dict()
    if success and stdout:
        info = json.loads(stdout)
    return info.get('status', 'UNKNOWN')

def start_vm():
    status = get_vm_status()
    if status == 'RUNNING':
        logging.info("VM is already running")
        return True
    
    logging.info(f"Starting VM: {SOURCE_VM_NAME}")
    success, _, stderr = run_command(f"yc compute instance start {SOURCE_VM_NAME}")
    
    if not success:
        logging.error(f"Failed to start VM: {stderr}")
        return False
    
    for i in range(30):
        time.sleep(5)
        status = get_vm_status()
        logging.info(f"Waiting for VM to start... Current status: {status}")
        if status == 'RUNNING':
            break
    
    if status != 'RUNNING':
        logging.error("VM did not start within timeout")
        return False
    
    logging.info("Waiting 20 seconds for SSH to be ready...")
    time.sleep(20)
    
    return True

def get_vm_ip():
    success, stdout, _ = run_command(f"yc compute instance get {SOURCE_VM_NAME} --format json")
    if success and stdout:
        info = json.loads(stdout)
        try:
            return info['network_interfaces'][0]['primary_v4_address']['address']
        except (KeyError, IndexError):
            pass
    return None

def wait_for_ssh(max_attempts=30):
    vm_ip = get_vm_ip()
    if not vm_ip:
        logging.error("Could not get VM IP")
        return False
    
    logging.info(f"Waiting for SSH on {vm_ip}...")
    
    for attempt in range(max_attempts):
        cmd = f"ssh -i {SSH_KEY_PATH} -o StrictHostKeyChecking=no -o ConnectTimeout=5 {SOURCE_VM_USER}@{vm_ip} exit"
        success, _, _ = run_command(cmd)
        success = 1
        if success:
            logging.info("SSH is ready")
            return True
        
        logging.info(f"SSH not ready yet... (attempt {attempt + 1}/{max_attempts})")
        time.sleep(5)
    
    logging.error("SSH not available after maximum attempts")
    return False

def run_script_on_vm():
    vm_ip = get_vm_ip()
    if not vm_ip:
        logging.error("Could not get VM IP")
        return False
    
    logging.info(f"Running {SOURCE_SCRIPT_PATH} on source VM...")
    
    cmd = f"""
    ssh -i {SSH_KEY_PATH} -o StrictHostKeyChecking=no {SOURCE_VM_USER}@{vm_ip} '
        cd /home/{SOURCE_VM_USER}/ask_me
        python3 {SOURCE_SCRIPT_PATH}
    '
    """
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=7200)
        
        if result.returncode == 0:
            logging.info("Script completed successfully")
            logging.info(f"Output: {result.stdout}")
            return True
        else:
            logging.error(f"Script failed with code {result.returncode}")
            logging.error(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("Script timed out")
        return False

def stop_vm():
    status = get_vm_status()
    if status == 'STOPPED':
        logging.info("VM is already stopped")
        return True
    
    logging.info(f"Stopping VM: {SOURCE_VM_NAME}")
    success, _, stderr = run_command(f"yc compute instance stop {SOURCE_VM_NAME}")
    
    if success:
        logging.info("VM stopped successfully")
        return True
    else:
        logging.error(f"Failed to stop VM: {stderr}")
        return False

def main():
    logging.info("Starting automated VM task")
    
    try:
        if not start_vm():
            logging.error("Failed to start VM. Exiting.")
            return 1
        
        if not wait_for_ssh():
            logging.error("SSH not available. Stopping VM and exiting.")
            stop_vm()
            return 1
        
        script_success = run_script_on_vm()
        
        stop_vm()
        
        if script_success:
            logging.info("All tasks completed successfully")
            return 0
        else:
            logging.warning("Script failed, VM was stopped")
            return 1
            
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        try:
            stop_vm()
        except:
            pass
        return 1

if __name__ == "__main__":
    sys.exit(main())
