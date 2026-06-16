import os, sys
from pathlib import Path

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None
    
from horizondrive.schemas.args import Args
from horizondrive.trainer import WanUnifiedTrainer

def main():
    args = Args.parse_args()
    if args.debugpy:
        import debugpy
        debugpy.listen(15678)
        print("Waiting for debugger to attach...")

    trainer = WanUnifiedTrainer(args)
    trainer.eval()


if __name__ == "__main__":
    main()
