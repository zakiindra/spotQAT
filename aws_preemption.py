import os
import sys
import time
import subprocess
import numpy as np
import argparse
import pandas as pd

def sample_aws_lifetime(cdf_csv_path="Emulator-unified/us-east-1a_cdf.csv"):
    csv_path = os.path.join(os.path.dirname(__file__), cdf_csv_path)
    
    # Read the empirical CDF from the CSV file
    df = pd.read_csv(csv_path)
    
    # The CDF file may contain duplicate CDF values for different durations, 
    # we group by CDF and get the mean duration or simply drop duplicates to make it strictly increasing for interpolation
    df = df.drop_duplicates(subset=['CDF'])
    
    # Sample a uniform random variable
    u = np.random.uniform(0, 1)
    
    # Interpolate to find the duration corresponding to the sampled probability
    sampled_sec = np.interp(u, df['CDF'], df['Duration'])
    
    return sampled_sec

def main():
    parser = argparse.ArgumentParser(description="AWS Preemption Simulator")
    parser.add_argument("--dry-run", action="store_true", help="Print simulated lifetime and exit")
    args = parser.parse_args()

    print("Initializing AWS Preemption Simulator...")
    lifetime = sample_aws_lifetime()
    print(f"Simulated AWS spot lifetime: {lifetime:.2f} seconds ({lifetime/3600:.2f} hours)")
    
    if args.dry_run:
        return
        
    # Running the modified training script as requested
    script_to_run = os.path.join(os.path.dirname(__file__), "train_and_qat_modified.py")
    
    print(f"Launching {script_to_run} ...")
    process = subprocess.Popen([sys.executable, script_to_run])
    
    start_time = time.time()
    
    try:
        while True:
            ret_code = process.poll()
            if ret_code is not None:
                print(f"\nProcess finished gracefully with exit code {ret_code}.")
                break
                
            elapsed = time.time() - start_time
            if elapsed >= lifetime:
                print(f"\n[!] Preemption triggered at {elapsed:.2f}s! Revoking capacity.")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print("Process hard-killed to simulate Spot Instance interruption.")
                process.wait()
                break
                
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nOrchestrator interrupted. Killing child training process.")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()
