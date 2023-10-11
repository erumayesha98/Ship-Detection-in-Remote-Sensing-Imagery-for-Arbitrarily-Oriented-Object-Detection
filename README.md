# Ship-Detection-in-Remote-Sensing-Imagery-for-Arbitrarily-Oriented-Object-Detection


This project provides a deep learning-based framework for ship detection in remote sensing imagery, with a focus on arbitrarily oriented objects. The framework is based on the YOLOv8 and UNET architectures, which are both state-of-the-art models for object detection and segmentation, respectively.

To train the framework on Google Colab, you will need to:

Create a new Colab notebook.
Install the necessary dependencies:
!pip install -r requirements.txt
Upload the Airbus Ship Detection Challenge dataset to your Colab drive.
Mount your Colab drive:
from google.colab import drive
drive.mount('/content/drive')
Set the dataset_dir variable to point to the directory on your Colab drive where the dataset is located.
Set the output_dir variable to point to the directory on your Colab drive where you want to save the trained model.
To train the YOLOv8 model, run the following command:

!python train_yolov8.py --dataset_dir /content/drive/MyDrive/Airbus_Ship_Detection_Challenge --output_dir /content/drive/MyDrive/yolov8_model

To train the UNET model, run the following command:

!python train_unet.py --dataset_dir /content/drive/MyDrive/Airbus_Ship_Detection_Challenge --output_dir /content/drive/MyDrive/unet_model

Once trained, you can use the models to detect ships in new remote sensing images. To do this, simply run the following command:

!python test_yolov8.py --image_path /content/drive/MyDrive/image.jpg --output_path /content/drive/MyDrive/yolov8_bounding_boxes.txt

or

!python test_unet.py --image_path /content/drive/MyDrive/image.jpg --output_path /content/drive/MyDrive/unet_segmentation_mask.png

The image_path argument should point to the image that you want to detect ships in. The output_path argument should point to the directory where you want to save the output bounding boxes or segmentation masks.

**Results**
The following table shows the results achieved by the YOLOv8 and UNET models on the Airbus Ship Detection Challenge dataset, when trained and executed on Google Colab:

Metric	YOLOv8	UNet
mAP	0.888	0.8
Precision	0.801	0.95
Recall	0.893	0.75
Speed	6.7ms/image	85ms/image


As you can see, the YOLOv8 model is faster than the UNET model, but the UNET model is slightly more accurate. Ultimately, the best model to use will depend on your specific needs. If speed is important, then the YOLOv8 model is a good choice. If accuracy is more important, then the UNET model is a better choice.

Tips
If you are running out of memory on Google Colab, you can try reducing the batch size or using a smaller image size.
You can also try using a different GPU type, such as the Tesla K80 or Tesla P100.
If you are training the YOLOv8 model, you can try using the --pretrained_weights option to load pre-trained weights from a different model. This can help to improve the training speed and accuracy.
Contact
If you have any questions or feedback, please feel free to contact me at [email protected]
