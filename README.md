# Scripts for training Layout Detection Models using Detectron2

## Usage

### Directory Structure

- In `tools/`, we provide a series of handy scripts for converting data formats and training the models.
- In `scripts/`, it lists specific command for running the code for processing the given dataset. 
- The `configs/` contains the configuration for different deep learning models, and is organized by datasets.

### How to train the models? 

- Get the dataset and annotations -- if you are not sure, feel free to check [this tutorial](https://github.com/Layout-Parser/layout-parser/tree/main/examples/Customizing%20Layout%20Models%20with%20Label%20Studio%20Annotation). 
- Duplicate and modify the config files and training scripts
    - For example, you might want to copy [`configs/prima/fast_rcnn_R_50_FPN_3x`](configs/prima/fast_rcnn_R_50_FPN_3x.yaml) to [`configs/your-dataset-name/fast_rcnn_R_50_FPN_3x`](configs/prima/fast_rcnn_R_50_FPN_3x.yaml), and you can create your own `scripts/train_<your-dataset-name>.sh` based on [`scripts/train_prima.sh`](scripts/train_prima.sh).
    - You'll modify the `--dataset_name`, `--json_annotation_train`, `--image_path_train`, `--json_annotation_val`, `--image_path_val`, and `--config-file` args appropriately. 
- If you have a dataset with segmentation masks, you can try to train with the [`mask_rcnn model`](configs/prima/mask_rcnn_R_50_FPN_3x.yaml); otherwise you might want to start with the [`fast_rcnn model`](configs/prima/fast_rcnn_R_50_FPN_3x.yaml)
    - If you see error `AttributeError: Cannot find field 'gt_masks' in the given Instances!` during training, this means you should not use 

## Supported Datasets

- Prima Layout Analysis Dataset [`scripts/train_prima.sh`](https://github.com/Layout-Parser/layout-model-training/blob/master/scripts/train_prima.sh)
    - You will need to download the dataset from the [official website](https://www.primaresearch.org/dataset/) and put it in the `data/prima` folder. 
    - As the original dataset is stored in the [PAGE format](https://www.primaresearch.org/tools/PAGEViewer), the script will use [`tools/convert_prima_to_coco.py`](https://github.com/Layout-Parser/layout-model-training/blob/master/tools/convert_prima_to_coco.py) to convert it to COCO format. 
    - The final dataset folder structure should look like:
        ```bash
        data/
        └── prima/
            ├── Images/
            ├── XML/
            ├── License.txt
            └── annotations*.json
        ```

## Reference 

- **[cocosplit](https://github.com/akarazniewicz/cocosplit)**  A script that splits the coco annotations into train and test sets.
- **[Detectron2](https://github.com/facebookresearch/detectron2)** Detectron2 is Facebook AI Research's next generation software system that implements state-of-the-art object detection algorithms. 