import csv
import os
import sys
import time
import subprocess
import numpy as np
import argparse
import pandas as pd
import uuid

def sample_aws_lifetime(cdf_csv_path="data/aws/us-east-1a_cdf.csv"):
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
    parser.add_argument("--checkpointing-method", type=str, default="fixed", choices=["fixed", "async", "adaptive", "adaptive_async", "young_daly", "young_daly_async", "none"], help="Checkpointing method to pass to the training script")
    parser.add_argument("--max-sample-time", type=float, default=float('inf'), help="Maximum sample time in seconds")
    parser.add_argument("--sim_id", type=str, default=uuid.uuid4().hex[:8], help="Unique simulation ID")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA_VISIBLE_DEVICES ID")
    parser.add_argument("--num_epochs_fp", type=int, default=3, help="Number of FP epochs")
    parser.add_argument("--num_epochs_qat", type=int, default=3, help="Number of QAT epochs")
    parser.add_argument("--risk-threshold", type=float, default=0.05, help="Risk threshold for adaptive checkpointing")
    parser.add_argument("--window-size", type=int, default=600, help="Window size for adaptive checkpointing")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor for adaptive checkpointing")
    parser.add_argument("--min-interval", type=int, default=300, help="Minimum interval for adaptive checkpointing")
    parser.add_argument("--delta", type=float, default=60.0, help="Checkpoint write overhead in seconds")
    parser.add_argument("--mttf", type=float, default=3600.0, help="Mean time to failure in seconds")
    args = parser.parse_args()

    print("Initializing AWS Preemption Simulator...")
    script_to_run = os.path.join(os.path.dirname(__file__), "train_and_qat_modified.py")

    while True:
        lifetime = sample_aws_lifetime()
        while lifetime > args.max_sample_time:
            lifetime = sample_aws_lifetime()
            
        print(f"Simulated AWS spot lifetime: {lifetime:.2f} seconds ({lifetime/3600:.2f} hours)")
        
        if args.dry_run:
            if lifetime <= args.max_sample_time:
                break
                
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        
        print(f"Launching {script_to_run} with method {args.checkpointing_method} on GPU {args.gpu_id} (sim_id {args.sim_id})...")
        process = subprocess.Popen([
            sys.executable, script_to_run, 
            f"--checkpointing={args.checkpointing_method}", 
            f"--sim_id={args.sim_id}",
            f"--num_epochs_fp={args.num_epochs_fp}",
            f"--num_epochs_qat={args.num_epochs_qat}",
            f"--max_sample_time={args.max_sample_time}",
            f"--risk_threshold={args.risk_threshold}",
            f"--window_size={args.window_size}",
            f"--scale_factor={args.scale_factor}",
            f"--min_interval={args.min_interval}",
            f"--delta={args.delta}",
            f"--mttf={args.mttf}"
        ], env=env)
        
        start_time = time.time()
        
        try:
            while True:
                ret_code = process.poll()
                if ret_code is not None:
                    print(f"\nProcess finished gracefully with exit code {ret_code}.")
                    return
                    
                elapsed = time.time() - start_time
                if elapsed >= lifetime:
                    # Construct the path to the same timing log used by the trainer
                    log_path = os.path.join(f"./qat_experiment_out/{args.sim_id}", "timing_log.csv")
                    
                    with open(log_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        # Manually append the eviction event with timestamp 
                        # Format: timestamp, model, run_type, phase, epoch, step, action, duration, risk
                        writer.writerow([time.time(), "N/A", "spot", "N/A", "N/A", "N/A", "eviction_triggered", 0.0, 1.0])

                    print(f"\n[!] Preemption triggered at {elapsed:.2f}s! Revoking capacity.")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    print("Process hard-killed to simulate Spot Instance interruption. Restarting on new instance...")
                    process.wait()
                    break
                    
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nOrchestrator interrupted. Killing child training process.")
            process.terminate()
            process.wait()
            return

if __name__ == "__main__":
    main()
