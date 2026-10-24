from multiprocessing import freeze_support

from ann_benchmarks.main import main

if __name__ == "__main__":
    freeze_support()
    import sys
    if "--max-n-algorithms" in sys.argv:
        i = sys.argv.index("--max-n-algorithms")
        del sys.argv[i:i + 2]
    main()
