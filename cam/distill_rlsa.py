"""Entry point of stage 1 of the distillation warm start (local addition, see cam/UPSTREAM.md).

Takes main.py's command line with --method CAMPrior (the teacher is set with CAMPrior's --prior_*
flags) and trains solvers/rlsa_distill.RLSADistill instead of CAMPrior. main.py itself is not
changed, so runs and evaluations that use it are unaffected.

    accelerate launch distill_rlsa.py <main.py flags> --method CAMPrior --prior_tau0 1 --do_train ...

Stage 2 is plain CAM: main.py --method CAM ... --checkpoint <this run's save_dir>/best.pt
"""
import sys

import main
from solvers.rlsa_distill import RLSADistill


def run(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    methods = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == '--method']
    if not methods or methods[-1] != 'CAMPrior':
        raise SystemExit('distill_rlsa.py: pass --method CAMPrior (the teacher is the CAMPrior prior)')
    main.CAMPrior = RLSADistill      # main.main() builds the solver of --method CAMPrior by this name
    sys.argv = [sys.argv[0]] + list(argv)
    main.main()


if __name__ == '__main__':
    run()
