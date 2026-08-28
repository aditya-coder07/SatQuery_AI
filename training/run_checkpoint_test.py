import subprocess
import time
import os
import shutil
import sys

def run_test():
    ckpt_dir = "checkpoints"
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("--- Starting initial training run ---")
    script_path = os.path.join("training", "checkpoint_resume_test.py")
    
    proc = subprocess.Popen(
        [sys.executable, script_path, "--ckpt-dir", ckpt_dir, "--save-every", "5", "--total-steps", "100"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    saved_checkpoint = False
    
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        print(line, end="")
        
        if "Saved checkpoint to" in line:
            saved_checkpoint = True
            time.sleep(0.5)
            print("--- Simulating free-tier disconnect (SIGKILL) ---")
            proc.kill()
            break
            
    proc.wait()
    
    if not saved_checkpoint:
        print("ERROR: Process exited before saving a checkpoint.")
        sys.exit(1)

    print("--- Resuming training run ---")
    proc2 = subprocess.Popen(
        [sys.executable, script_path, "--ckpt-dir", ckpt_dir, "--resume", "--save-every", "5", "--total-steps", "15"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    resume_step = -1
    for line in proc2.stdout:
        print(line, end="")
        if "RESUMING TRAINING FROM STEP" in line:
            resume_step = int(line.strip().split()[-1])
            
    proc2.wait()
    
    if resume_step > 0:
        print(f"\nSUCCESS: Training resumed correctly from step {resume_step} rather than 0.")
        sys.exit(0)
    else:
        print("\nERROR: Failed to resume correctly.")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
