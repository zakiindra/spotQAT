import json
import csv
import os
import sys
import time
import subprocess
import numpy as np
import argparse
import uuid
try:
    from scipy.optimize import curve_fit, root_scalar
except ImportError:
    print("Scipy is not installed. Please run `uv add scipy` or `pip install scipy`.")
    sys.exit(1)


# Preemption based on Prateek paper in references/preemption/paper.txt

def cdf_model(t, A, tau1, tau2, b):
    # F(t) = A * (1 - e^{-t/t1} + e^{(t-b)/t2})
    return A * (1 - np.exp(-t/tau1) + np.exp((t-b)/tau2))

def fit_and_sample_lifetime(data_path="data/gcp/data.json"):
    data_path = os.path.join(os.path.dirname(__file__), data_path)
    
    with open(data_path, "r") as f:
        data = json.load(f)
        
    lifetimes_sec = []
    for key, val in data.items():
        if "time_in_sec" in val:
            lifetimes_sec.append(val["time_in_sec"])
            
    # Convert to hours
    lifetimes_hours = np.array(lifetimes_sec) / 3600.0
    lifetimes_hours = np.sort(lifetimes_hours)
    
    # Calculate empirical CDF
    n = len(lifetimes_hours)
    empirical_cdf = np.arange(1, n + 1) / n
    
    # Initial guesses based on the paper
    p0 = [0.45, 1.0, 0.8, 24.0]
    
    try:
        popt, _ = curve_fit(cdf_model, lifetimes_hours, empirical_cdf, p0=p0, bounds=([0, 0, 0, 0], [1, 10, 10, 30]), maxfev=10000)
        A, tau1, tau2, b = popt
        print(f"Fitted parameters: A={A:.4f}, tau1={tau1:.4f}, tau2={tau2:.4f}, b={b:.4f}")
    except Exception as e:
        print(f"Failed to fit CDF model: {e}. Falling back to default parameters.")
        A, tau1, tau2, b = p0
        
    # Sample a lifetime using inverse transform sampling
    u = np.random.uniform(0, 1)
    
    def obj(t):
        return cdf_model(t, A, tau1, tau2, b) - u
        
    try:
        res = root_scalar(obj, bracket=[0, 24], method='brentq')
        sampled_hours = res.root
    except ValueError:
        if u > cdf_model(24, A, tau1, tau2, b):
            sampled_hours = 24.0
        else:
            sampled_hours = 0.0
            
    sampled_sec = sampled_hours * 3600.0
    return sampled_sec

def main():
    parser = argparse.ArgumentParser(description="Google Preemption Simulator")
    parser.add_argument("--dry-run", action="store_true", help="Print simulated lifetime and exit")
    parser.add_argument("--checkpointing-method", type=str, default="fixed", choices=["fixed", "async", "adaptive", "young_daly", "adaptive_async", "young_daly_async", "none"], help="Checkpointing method to pass to the training script")
    parser.add_argument("--max-sample-time", type=float, default=float('inf'), help="Maximum sample time in seconds")
    parser.add_argument("--sim_id", type=str, default=uuid.uuid4().hex[:8], help="Unique simulation ID")
    parser.add_argument("--gpu_id", type=str, default="0", help="CUDA_VISIBLE_DEVICES ID")
    parser.add_argument("--num_epochs_fp", type=int, default=3, help="Number of FP epochs")
    parser.add_argument("--num_epochs_qat", type=int, default=3, help="Number of QAT epochs")
    parser.add_argument("--risk-threshold", type=float, default=0.05, help="Risk threshold for adaptive checkpointing")
    parser.add_argument("--window-size", type=int, default=600, help="Window size for adaptive checkpointing")
    parser.add_argument("--scale-factor", type=float, default=1.0, help="Scale factor for time compression")
    parser.add_argument("--min-interval", type=float, default=300, help="Minimum interval between checkpoints")
    parser.add_argument("--delta", type=float, default=60.0, help="Checkpoint write overhead in seconds")
    parser.add_argument("--mttf", type=float, default=3600.0, help="Mean time to failure in seconds")
    args = parser.parse_args()

    print("Initializing Google Preemption Simulator...")
    script_to_run = os.path.join(os.path.dirname(__file__), "train_and_qat_modified.py")

    while True:
        lifetime = fit_and_sample_lifetime()
        while lifetime > args.max_sample_time:
            lifetime = fit_and_sample_lifetime()
            
        print(f"Simulated spot lifetime: {lifetime:.2f} seconds ({lifetime/3600:.2f} hours)")
        
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
