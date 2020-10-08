import json
import os
import torch
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.argoverse_datamodule import ArgoverseDataModule
from src.data.dummy_datamodule import DummyDataModule
from src.models.WIMP import WIMP


def parse_arguments():
    parser = ArgumentParser()

    # Load experiment and trainer-sepcific args
    parser = add_experimental_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)

    # Parse dataset model to use
    parser.add_argument('--dataset', type=str, default='argoverse', help='Name of dataset to use')
    parser.add_argument('--model-name', type=str, default='WIMP', help='Name of model to load')
    temp_args, _ = parser.parse_known_args()

    # Load dataset specific args
    if temp_args.dataset == 'argoverse':
        parser = ArgoverseDataModule.add_data_specific_args(parser)
    else:
        raise NotImplementedError

    # Load model specific args
    if temp_args.model_name == 'WIMP':
        parser = WIMP.add_model_specific_args(parser)
    else:
        raise NotImplementedError

    args = parser.parse_args()
    # with open('args.json', 'r') as f:
    #     args.__dict__ = json.load(f)
    return args


def add_experimental_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # General Params
    parser.add_argument("--mode", required=True, type=str, choices=['train', 'val', 'trainval', 'oracle-val'],
                        help='Mode to run forecasting model in')
    parser.add_argument('--seed', type=int, help="Seed the random parameter generation")

    # I/O and Feature Space Params
    parser.add_argument('--predict-delta', action='store_true', help="Predict delta-xy coordinates instead of absolute") # NOQA
    parser.add_argument('--IFC', action='store_true', help="Compute centerline features when predicting trajectory") # NOQA
    parser.add_argument('--map-features', action='store_true', help="Compute map features") # NOQA
    parser.add_argument('--no-heuristic', action='store_true', help="Don't use heuristic centerline features") # NOQA
    parser.add_argument('--use-oracle', action='store_true', help='Whether to use features obtained from oracle') # NOQA

    # Datamodule Params
    parser.add_argument('--dataroot', required=True, help="Path to the processed dataset folder")
    parser.add_argument("--batch-size", type=int, default=25, help="Training batch size")
    parser.add_argument('--workers', type=int, default=8, help="Number of dataloader workers")

    # Trainer Params
    parser.add_argument("--gpus", type=int, default=1, help='# of GPUs to use for training')
    parser.add_argument("--check-val-every-n-epoch", type=int, default=3, help="# of training epochs between val") # NOQA
    parser.add_argument("--max-epochs", type=int, default=120, help="Max # of training epochs")
    parser.add_argument("--early-stop-threshold", type=int, default=5, help="Number of consecutive val epochs without improvement before termination") # NOQA
    parser.add_argument('--distributed-backend', default=None, help='Trainer backend')
    parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes used')
    parser.add_argument('--precision', default=32, type=int, help='Precision employed in weights')
    parser.add_argument('--resume-from-checkpoint', help='Path to checkpoint to resume training from')

    # Logging Params
    parser.add_argument('--experiment-name', type=str, help='Save file prefix')

    return parser


def cli_main(args):
    print(args)

    # Set global random seed
    pl.seed_everything(args.seed)

    # Initialize data module
    dm = ArgoverseDataModule(args)

    # Initialize selected model
    if args.model_name == 'WIMP':
        model = WIMP(args)

    # Initialize trainer
    logger = TensorBoardLogger(os.getcwd(), name='experiments', version=args.experiment_name)
    early_stop_cb = EarlyStopping(patience=args.early_stop_threshold, verbose=True)
    trainer = pl.Trainer(gpus=args.gpus, check_val_every_n_epoch=args.check_val_every_n_epoch,
                         max_epochs=args.max_epochs, default_root_dir=os.getcwd(),
                         distributed_backend=args.distributed_backend, num_nodes=args.num_nodes,
                         precision=args.precision, resume_from_checkpoint=args.resume_from_checkpoint,
                         logger=logger, callbacks=[early_stop_cb])
    trainer.fit(model, dm)


if __name__ == '__main__':
    args = parse_arguments()
    cli_main(args)
