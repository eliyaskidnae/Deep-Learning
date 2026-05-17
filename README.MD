# Deep-Learning — Project Summary

## Object Detection (Custome_Object_Detection/object_detection.ipynb)

This notebook implements a simplified custom object-detection pipeline and compares a small custom CNN-based detector with a YOLO model (Ultralytics YOLOv8). It focuses on dataset creation, model design, training, evaluation and comparison.

Key points:
- Purpose: create a synthetic dataset by pasting single objects (3 chosen classes) onto background images, train a custom PyTorch object detector, fine-tune/evaluate a YOLO model, and compare metrics.
- Objects used (example): car, traffic-crossing sign, motorcycle/stop/two_way (the notebook shows variants).
- Dataset: synthetic images are created by inserting a single object into random background images and saving bounding boxes in YOLO (cx, cy, w, h) format. The notebook creates train/val/test splits and wraps data into a PyTorch `Dataset`.
- Custom model: a small CNN backbone with 5 convolutional blocks (batch-norm, ReLU, dropout, max-pool), a shared fully-connected head and two branches: classification (num_classes) and bbox regression (4 values). A VGG16-based variant is provided as a suggested improvement.
- Training: classification loss (CrossEntropy) + bounding-box loss (SmoothL1); optimizer is Adam; learning rate scheduling and checkpointing (WandB logging supported). Training and validation loops, early stopping, and checkpoint save/load utilities are included.
- Evaluation: IoU calculation, mAP (using torchmetrics MeanAveragePrecision with cxcywh format), visualizations of predicted vs. ground-truth boxes, and side-by-side comparison with YOLO outputs.

How to run (high-level):
1. Prepare data directories: `objects/` (object PNGs with alpha), `background/` (background images). The notebook contains example crawling code using `icrawler` and data-creation code which generates datasets under `datasets/{train,val,test}`.
2. Install Python dependencies used in the notebook: PyTorch, torchvision, icrawler, ultralytics (YOLO), torchmetrics, wandb (if logging). The notebook includes pip commands.
3. Run the dataset creation cells to generate labeled images and text label files in YOLO format.
4. Instantiate `ObjectDetectionDataset`, create DataLoaders, build `CustomObjectDetectionModel`, then train via the `train_model` function. Optionally fine-tune YOLO with `yolo task=detect mode=train ...` (notebook uses `!yolo`).
5. Evaluate both models with `evaluate_test_set` (custom) and YOLO's `.val()` API. Visualize predictions using the `visualize_predictions`, `yolo_detection`, and comparison plotting functions.

Notes and tips:
- The synthetic dataset uses single-object images, so metrics like mAP@0.50, mAP@0.75, IoU and visual inspection are appropriate.
- Use early stopping, learning-rate scheduling and checkpointing to prevent runtime issues on limited compute (e.g., Colab).
- Many helper utilities are included (checkpointing, training loops, plotting).

## Image Captioning (Image Captioning/Image_Captioning.ipynb)

This notebook implements an encoder-attention-decoder image-captioning system trained on the Flickr8k dataset and compares the custom model to a pretrained transformer-based model (BLIP).

Key points:
- Purpose: build an Encoder (ResNet backbone) + Attention + Decoder (LSTM) model, train on Flickr8k, visualize attention maps, and evaluate using BLEU scores.
- Dataset: Flickr8k captions are tokenized, a vocabulary is created (with PAD, SOS, EOS, UNK tokens), and captions are vectorized. Data is split into train/val/test and loaded with a custom collate function that pads varying-length captions.
- Custom model: Encoder (ResNet-50 features), Attention module (linear transforms producing attention weights), and an LSTM-based decoder that consumes embeddings + attention context to predict the next token.
- Training: CrossEntropy loss (ignore PAD token), optimizer Adam, learning-rate schedule, checkpoint saving/loading. Training and validation loops are included, with gradient clipping and optional early stopping.
- Evaluation: BLEU-1/2/3 scores on the test set; attention visualization functions overlay the attention weights on images to interpret the model's focus.
- Comparison model: BLIP (Salesforce/blip-image-captioning-base) is loaded via Hugging Face Transformers and used to generate captions for comparison.

How to run (high-level):
1. Download Flickr8k dataset and place images under `flickr8k/Images` and captions in `flickr8k/captions.txt` (the notebook shows Kaggle steps).
2. Install required packages (torch, torchvision, transformers, torchtext). The notebook contains pip install commands for Colab compatibility.
3. Create `FlickrDataset`, build dataloaders (with `collate_fn`), and initialize the `EncoderDecoder` model.
4. Train with the provided `train_model` function and monitor losses; save checkpoints.
5. Generate captions with `DecoderRNN.generate_caption`, visualize attention via `plot_attention`, and compute BLEU scores with `torchtext.data.metrics.bleu_score` for the custom model and BLIP.

Notes and tips:
- The notebook emphasizes attention visualizations (required for full credit) and saving checkpoints to resume training on platforms like Colab.
- BLIP evaluation reuses the same metric (BLEU) for a fair comparison; the notebook includes sample BLEU score outputs and a short analysis.
---

