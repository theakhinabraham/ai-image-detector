import torch

from src.gradcam import GradCAM
from src.models import ResNet18Detector, SmallCNN


def test_smallcnn_output_shape():
    out = SmallCNN().eval()(torch.randn(4, 3, 32, 32))
    assert out.shape == (4,)  # one logit per image


def test_gradcam_shape_and_range_smallcnn():
    model = SmallCNN().eval()
    cam, logit = GradCAM(model, model.gradcam_layer())(torch.randn(1, 3, 32, 32))
    assert cam.shape == (32, 32)
    assert cam.min() >= 0.0 and cam.max() <= 1.0
    assert isinstance(logit, float)


def test_gradcam_shape_resnet18():
    model = ResNet18Detector(pretrained=False).eval()  # no download needed
    cam, _ = GradCAM(model, model.gradcam_layer())(torch.randn(1, 3, 64, 64))
    assert cam.shape == (64, 64)
