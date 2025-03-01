import PIL.Image

from yolox.models import YOLOX, YOLOXProcessor

class TestDetections:
    def test_detections(self) -> None:
        test_img = PIL.Image.open('/Users/asiegel/Dropbox/workspace/pixeltable/pixeltable/docs/resources/images/000000000009.jpg')
        model = YOLOX.from_pretrained('yolox_l')
        processor = YOLOXProcessor('yolox_l')
        tensor = processor([test_img])
        output = model(tensor)
        results = processor.postprocess([test_img], output)
        print(results)
