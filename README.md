
# EasyOCR Arabic Fine-Tuning

This repository is designed to provide a streamlined process for fine-tuning the EasyOCR model specifically for Arabic datasets.

## Environment Setup

Before you begin, ensure that your system has all the necessary dependencies installed.

### Install Dependencies

Run the following command in your terminal to install the required libraries:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

To fine-tune the model with a new dataset, format your data and convert it into an LMDB database for efficient training.

### Format the Dataset

Store your dataset images in a single folder and create a corresponding `Labels.txt` file containing mappings from image filenames to their textual labels.

## Label Format Example

Your `Labels.txt` should have each line representing an image and its label in the following format:

```
image1.jpg text_of_image_1
image2.jpg text_of_image_2
...
```

This format ensures that each image is correctly associated with its corresponding text label for the training process.

### Create LMDB Database

Run the following command to convert your image files and labels into an LMDB format database:

```bash
python create_lmdb_dataset.py --inputPath ./dataset/ --gtFile ./dataset/Labels.txt --outputPath ./dataset/lmdb_output
```

## Model Training

Configure your training parameters and execute the training command.

### Training Command

Here's an example command to start training:

```bash
python train.py --train_data ./dataset/lmdb_output --valid_data ./dataset/lmdb_output --select_data "/" --batch_ratio 1.0 --Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --batch_size 2 --data_filtering_off --workers 0 --batch_max_length 80 --num_iter 10 --valInterval 5 --saved_model ./saved_models/arabic.pth
```

## Running Inference

After training, use the trained model to predict new images.

### Inference Command

```bash
python demo.py --Transformation None --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction CTC --image_folder <path_to_test_images> --saved_model ./saved_models/arabic.pth
```

## GPU and CPU Configuration

Depending on whether you are using a GPU or CPU, you may need to adjust the model loading code in `train.py`:

- **For GPU users:** Uncomment lines 86-88 and comment out lines 93-95.
- **For CPU users:** Uncomment lines 93-95 and comment out lines 86-88.

## Conclusion

This setup enables teams to efficiently prepare, train, and deploy OCR models for Arabic text recognition. Follow these steps to ensure a successful implementation.


