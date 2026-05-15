import subprocess
import json
import time
import os

def run_bench(command, args):
    full_cmd = ["./target/release/hms-ann-bench", command] + args
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None
    try:
        # Find the JSON line in output
        for line in result.stdout.split('\n'):
            if line.strip().startswith('{'):
                return json.loads(line)
    except:
        print(f"Parse error: {result.stdout}")
    return None

def main():
    print("HMS SOTA Scaling Benchmark")
    print("==========================")
    
    n_values = [1000, 10000, 25000] 
    results = []
    
    db_path = "benchmarks/data/scaling_test"

    for n in n_values:
        if os.path.exists(db_path):
            subprocess.run(["rm", "-rf", db_path])
            
        print(f"Testing N={n}...")
        
        # Load
        load_res = run_bench("load", ["--path", db_path, "--count", str(n)])
        
        # Query
        query_res = run_bench("query", ["--path", db_path, "--queries", "200"])
        
        if load_res and query_res:
            results.append({
                "n": n,
                "load_qps": load_res["throughput_ops_sec"],
                "p95_us": query_res["p95_us"]
            })

    print("\nResults Summary (JSON for Plotting):")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
