"""Explicit opt-in core training; never download datasets or regenerate splits."""
import argparse
import copy
from glaucoma.models.registry import CORE_APPLICATIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, help='Previously saved session.json')
    parser.add_argument('--metadata', help='Override frozen metadata CSV path')
    parser.add_argument('--model', choices=list(CORE_APPLICATIONS), required=True)
    parser.add_argument('--stage', choices=['stage1','stage2','both'], default='stage1')
    parser.add_argument('--weights', choices=['imagenet','none'], default='imagenet')
    parser.add_argument('--enable-training', action='store_true')
    parser.add_argument('--allow-imagenet-download', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Resume selected single stage from last complete epoch')
    args=parser.parse_args()
    if not args.enable_training:
        parser.error('Training disabled. Add --enable-training only when deliberately ready to fit.')
    if args.resume and args.stage=='both':
        parser.error('Resume a single explicit stage, not both.')
    from glaucoma.utils.notebook_runtime import load_session
    from glaucoma.models.factory import build_model
    from glaucoma.training.trainer import train_stage, select_best_checkpoint
    from glaucoma.utils.reproducibility import set_random_seed
    cfg=copy.deepcopy(load_session(args.session))
    cfg['model']['selected']=args.model
    cfg['model']['weights']=None if args.weights=='none' else 'imagenet'
    if args.metadata:
        cfg['metadata']['output']=args.metadata
    stages=['stage1','stage2'] if args.stage=='both' else [args.stage]
    for stage in stages:
        model=None
        if stage=='stage1' and not args.resume:
            if args.weights=='imagenet' and not args.allow_imagenet_download:
                parser.error('Add --allow-imagenet-download or choose --weights none for offline experiments.')
            set_random_seed(cfg['seed'])
            model=build_model(args.model,weights=cfg['model']['weights'])
        result=train_stage(cfg,stage,model=model,enabled=True,resume=args.resume)
        print(result)
    print(select_best_checkpoint(cfg,save_selection=True))


if __name__=='__main__':
    main()
