# Scripts for training Layout Detection Models using Detectron2

## Usage

- In `tools/`, we provide a series of handy scripts for converting data formats and training the models.
- In `scripts/`, it lists specific command for running the code for processing the given dataset. 
- The `configs/` contains the configuration for different deep learning models, and is organized by datasets.

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