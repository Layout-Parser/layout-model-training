# Modified based on https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py

import json
import argparse
import funcy
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description="Splits COCO annotations file into training and test sets."
)
parser.add_argument(
    "--annotation-path",
    metavar="coco_annotations",
    type=str,
    help="Path to COCO annotations file.",
)
parser.add_argument(
    "--train", type=str, help="Where to store COCO training annotations"
)
parser.add_argument("--test", type=str, help="Where to store COCO test annotations")
parser.add_argument(
    "--split-ratio",
    dest="split_ratio",
    type=float,
    required=True,
    help="A percentage of a split; a number in (0, 1)",
)
parser.add_argument(
    "--having-annotations",
    dest="having_annotations",
    action="store_true",
    help="Ignore all images without annotations. Keep only these with at least one annotation",
)


def save_coco(file, tagged_data):
    with open(file, "wt", encoding="UTF-8") as coco:
        json.dump(tagged_data, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    return funcy.lfilter(lambda a: int(a["image_id"]) in image_ids, annotations)


def main(
    annotation_path,
    split_ratio,
    having_annotations,
    train_save_path,
    test_save_path,
    random_state=None,
):

    with open(annotation_path, "rt", encoding="UTF-8") as annotations:
        coco = json.load(annotations)

    images = coco["images"]
    annotations = coco["annotations"]

    ids_with_annotations = funcy.lmap(lambda a: int(a["image_id"]), annotations)

    # Images with annotations
    img_ann = funcy.lremove(lambda i: i["id"] not in ids_with_annotations, images)
    tr_ann, ts_ann = train_test_split(
        img_ann, train_size=split_ratio, random_state=random_state
    )

    img_wo_ann = funcy.lremove(lambda i: i["id"] in ids_with_annotations, images)
    if len(img_wo_ann) > 0:
        tr_wo_ann, ts_wo_ann = train_test_split(
            img_wo_ann, train_size=split_ratio, random_state=random_state
        )
    else:
        tr_wo_ann, ts_wo_ann = [], []  # Images without annotations

    if having_annotations:
        tr, ts = tr_ann, ts_ann

    else:
        # Merging the 2 image lists (i.e. with and without annotation)
        tr_ann.extend(tr_wo_ann)
        ts_ann.extend(ts_wo_ann)

        tr, ts = tr_ann, ts_ann

    # Train Data
    coco.update({"images": tr, "annotations": filter_annotations(annotations, tr)})
    save_coco(train_save_path, coco)

    # Test Data
    coco.update({"images": ts, "annotations": filter_annotations(annotations, ts)})
    save_coco(test_save_path, coco)

    print(
        "Saved {} entries in {} and {} in {}".format(
            len(tr), train_save_path, len(ts), test_save_path
        )
    )


if __name__ == "__main__":
    args = parser.parse_args()

    main(
        args.annotation_path,
        args.split_ratio,
        args.having_annotations,
        args.train,
        args.test,
        random_state=24,
    )
