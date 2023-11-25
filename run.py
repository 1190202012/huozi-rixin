import argparse
import os
import warnings
import torch

from aaai24.config import Config

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # parse args
    parser = argparse.ArgumentParser(description='a project on retrieval enhance with controller')
    parser.add_argument('-c', '--config', type=str, default='config/base.yaml', help='config file(yaml) path')
    parser.add_argument('-d', '--debug', action='store_true',help='use valid dataset to debug your system')

    args, _ = parser.parse_known_args()
    
    if args.debug:
        import debugpy
        debugpy.connect(('192.168.1.50', 6789))
        debugpy.wait_for_client()
        debugpy.breakpoint()
        # import pydevd_pycharm
        # pydevd_pycharm.settrace('192.168.1.50', port=6789, stdoutToServer=True, stderrToServer=True)

    config = Config(args.config, debug=args.debug)
    
    if config["mode"] == "base":
        from aaai24.main import run_example, run_one_example
        
        run_example(config, debug=args.debug)
        # run_one_example(config, debug=args.debug)

    elif config["mode"] == "eval":
        from aaai24.eval import run_eval
        
        world_size = len(config["gpu"])
        
        torch.multiprocessing.spawn(run_eval, args=(config, args.debug), nprocs=world_size, join=True)

    elif config["mode"] == "demo":
        from aaai24.demo import build_demo

        demo = build_demo(config, debug=args.debug)
        demo.queue()
        demo.launch(share=True)

    else:
        assert False, "mode must in ['base', 'eval', 'demo']."
