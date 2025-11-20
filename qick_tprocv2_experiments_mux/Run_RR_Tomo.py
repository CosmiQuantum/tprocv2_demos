import subprocess
import time

def run_script(path):
    start = time.time()
    subprocess.run(["python", path], check=True)
    end = time.time()
    return end-start

def main():
    num_cycles = 3

    total_runtime = 0.0

    for cycle in range(1, num_cycles+1):
        print(f'Starting cycle {cycle}/{num_cycles}')
        cycle_start = time.time()

        print('Running RR')
        t1 = run_script('round_robin_benchmark.py')
        print(f'RR finished in {t1:.2f} sec')

        print('Running Tomography')
        t2 = run_script('mux_nexus_run_tomo.py')
        print(f'Tomography finished in {t2:.2f} sec')

        cycle_end = time.time()
        cycle_time = cycle_end - cycle_start
        total_runtime +=cycle_time

        print(f'Cycle {cycle} completed in {cycle_time:.2f} sec')
    print(f'\n All {num_cycles} cycles finished')
    print(f'Total runtime: {total_runtime:.2f} seconds')

if __name__ == "__main__":
    main()

#1RR on all 4 qubits: 676.7338342666626 (from RR), 678.04 (from this script)
#1Tomo on all 4 qubits: 366.36 (from this script)
