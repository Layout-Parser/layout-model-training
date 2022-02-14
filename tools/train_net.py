"""
The script is based on https://github.com/facebookresearch/detectron2/blob/master/tools/train_net.py. 
"""

import logging
import os
import json
from collections import OrderedDict
import detectron2.utils.comm as comm
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader

from detectron2.data.datasets import register_coco_instances

from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import pandas as pd


def get_augs(cfg):
    """Add all the desired augmentations here. A list of availble augmentations
    can be found here:
       https://detectron2.readthedocs.io/en/latest/modules/data_transforms.html
    """
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN,
            cfg.INPUT.MAX_SIZE_TRAIN,
            cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    horizontal_flip: bool = cfg.INPUT.RANDOM_FLIP == "horizontal"
    augs.append(T.RandomFlip(horizontal=horizontal_flip, vertical=not horizontal_flip))
    # Rotate the image between -90 to 0 degrees clockwise around the centre
    augs.append(T.RandomRotation(angle=[-90.0, 0.0]))
    return augs


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.

    Adapted from:
        https://github.com/facebookresearch/detectron2/blob/master/projects/DeepLab/train_net.py
    """

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=get_augs(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

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
        pd.DataFrame(res).to_csv(os.path.join(cfg.OUTPUT_DIR, "eval.csv"))
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    with open(args.json_annotation_train, "r") as fp:
        anno_file = json.load(fp)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(anno_file["categories"])
    del anno_file

    cfg.DATASETS.TRAIN = (f"{args.dataset_name}-train",)
    cfg.DATASETS.TEST = (f"{args.dataset_name}-val",)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # Register Datasets
    register_coco_instances(
        f"{args.dataset_name}-train",
        {},
        args.json_annotation_train,
        args.image_path_train,
    )

    register_coco_instances(
        f"{args.dataset_name}-val", 
        {}, 
        args.json_annotation_val, 
        args.image_path_val
    )
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
        pd.DataFrame(res).to_csv(f"{cfg.OUTPUT_DIR}/eval.csv")
        return res

    # Ensure that the Output directory exists
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

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
    parser.add_argument(
        "--dataset_name", 
        help="The Dataset Name")
    parser.add_argument(
        "--json_annotation_train",
        help="The path to the training set JSON annotation",
    )
    parser.add_argument(
        "--image_path_train",
        help="The path to the training set image folder",
    )
    parser.add_argument(
        "--json_annotation_val",
        help="The path to the validation set JSON annotation",
    )
    parser.add_argument(
        "--image_path_val",
        help="The path to the validation set image folder",
    )
    args = parser.parse_args()
    print("Command Line Args:", args)

    # Dataset Registration is moved to the main function to support multi-gpu training
    # See ref https://github.com/facebookresearch/detectron2/issues/253#issuecomment-554216517

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
