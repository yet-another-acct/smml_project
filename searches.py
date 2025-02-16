import sys
import argparse

def raise_(e): raise e

    
def main():    
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

    import lib.proj_script
    
    ap = argparse.ArgumentParser()
    ap.add_argument(
        'input', 
        type=str,
        help='''
input file with the same format as `your_dataset.csv` in this repository''')
    ap.add_argument(
        'output',
        type=str,
        help='''
Output destination for the results; assumed to exist and be a directory.''')
    ap.add_argument(
        '-c', '--max-concurrent',
        type=int,
        help='''
Maximum number of concurrent training runs. Default = cpu//2 + 2.
The main constraint is RAM, as kernelized algorithms take about 800M of memory.''')
    parsed = ap.parse_args()
    
    lib.proj_script.main(
        int(parsed.max_concurrent),
        parsed.output,
        parsed.input)

if __name__ == '__main__': main()