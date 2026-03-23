import subprocess
import time
import os
import requests
import sys
import yaml
sys.path.append(".")
import interactive

phonecall_path = "."
if phonecall_path not in sys.path:
    sys.path.insert(0, phonecall_path)

def setup_ollama_server():
    print("Launching Ollama server...")

    os.environ['OLLAMA_READ_TIMEOUT'] = '200'
    os.environ['OLLAMA_LOAD_TIMEOUT'] = '700'
    os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
    os.environ['OLLAMA_KEEP_ALIVE'] = '-1'
    #os.environ['OLLAMA_NUM_GPU'] = '999'  # Try to load all layers
    #os.environ['OLLAMA_GPU_LAYERS'] = '100'  # Explicitly set 80 layers
    os.environ['OLLAMA_MMAP'] = '0'  # Disable memory mapping (sometimes helps)

    os.environ['OLLAMA_GPU_OVERHEAD'] = '0'
    os.environ['OLLAMA_FLASH_ATTENTION'] = '1'

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['OLLAMA_CUDA'] = '1'
    #os.environ['OLLAMA_DEBUG'] = '0'

    server_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ
    )

    print("Waiting for server to start...")
    time.sleep(10)

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=30)
        if response.status_code == 200:
            print(" Ollama server is working")
        else:
            print(f" Ollama server responds with code {response.status_code}")
    except Exception as e:
        print(f" Ollama server NOT working: {e}")

def main():
    with open('config.yml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    setup_ollama_server()
    interactive.enhanced_interactive_mode(
        _model=config["llm_model"],
        csv_dir=config['folders']['csv_calls'],
        results_dir=config["folders"]["saved_results"],
        drive_path=None,
    )

if __name__ == '__main__':
    main()