import PIL.Image

from yolox.models import Yolox, YoloxProcessor

class TestDetections:
    def test_detections(self) -> None:
        test_imgs = [
            PIL.Image.open(f'/Users/asiegel/Dropbox/workspace/pixeltable/pixeltable/docs/resources/images/{file}')
            for file in ('000000000009.jpg', '000000000016.jpg', '000000000019.jpg')
        ]
        model = Yolox.from_pretrained('yolox_l')
        processor = YoloxProcessor('yolox_l')
        tensor = processor(test_imgs)
        output = model(tensor)
        results = processor.postprocess(test_imgs, output)
        print(results)
