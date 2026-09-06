# Model and image notices

This project fine-tunes timm pretrained weights:
- https://huggingface.co/timm/resnet18.a1_in1k
- https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k

Both model cards specify Apache-2.0. The exported model is a modified model:
the classification head has been replaced with three outputs and the model
has been fine-tuned on the documented Insects Commons 116 collection.
The selected architecture is recorded in model.json.
Apache-2.0 license text: LICENSE-APACHE-2.0.txt.

Photographs are credited individually in ../experiments/data/ATTRIBUTION.md.
The code's MIT license does not override image or pretrained-weight licenses.
