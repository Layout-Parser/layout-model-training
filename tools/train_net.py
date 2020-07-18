"""
The script is based on https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py. 
"""

import logging
import os
import json
from collections import OrderedDict
import torch
import sys 
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import pandas as pd

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Returns:
            DatasetEvaluator or None

        It is not implemented by default.
        """
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def eval_and_save(cls, cfg, model):
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        pd.DataFrame(res).to_csv(os.path.join(cfg.OUTPUT_DIR, 'eval.csv'))
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    with open(args.json_annotation_train, 'r') as fp:
        anno_file = json.load(fp)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(anno_file["categories"])
    del anno_file

    cfg.DATASETS.TRAIN = (f"{args.dataset_name}-train",)
    cfg.DATASETS.TEST = (f"{args.dataset_name}-val",)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
    
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)

        # Save the evaluation results
        pd.DataFrame(res).to_csv(f'{cfg.OUTPUT_DIR}/eval.csv')
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.eval_and_save(cfg, trainer.model))]
    )
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()

    # Extra Configurations for dataset names and paths
    parser.add_argument("--dataset_name",          default="", help="The Dataset Name")
    parser.add_argument("--json_annotation_train", default="", metavar="FILE", help="The path to the training set JSON annotation")
    parser.add_argument("--image_path_train",      default="", metavar="FILE", help="The path to the training set image folder")
    parser.add_argument("--json_annotation_val",   default="", metavar="FILE", help="The path to the validation set JSON annotation")
    parser.add_argument("--image_path_val",        default="", metavar="FILE", help="The path to the validation set image folder")

    args = parser.parse_args()
    print("Command Line Args:", args)
    
    # Register Datasets 
    dataset_name = args.dataset_name
    register_coco_instances(f"{dataset_name}-train", {}, 
                            args.json_annotation_train, 
                            args.image_path_train)

    register_coco_instances(f"{dataset_name}-val",   {}, 
                            args.json_annotation_val,   
                            args.image_path_val)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )