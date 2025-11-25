import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

def create_model(num_classes, pretrained=True, coco_model=False):
    # Load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights='DEFAULT'
    )
    
    anchor_generator = AnchorGenerator(
        sizes=((8, 16, 32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model.rpn.anchor_generator = anchor_generator

    if coco_model: # Return the COCO pretrained model for COCO classes.
        return model, coco_model
    
    # Get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(num_classes=81, pretrained=True, coco_model=True)
    summary(model)
