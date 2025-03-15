from PIL.Image import Image

from yolox.models import Yolox, YoloxProcessor


class TestDetections:
    def test_detections(self, test_images: list[Image]) -> None:
        model = Yolox.from_pretrained('yolox_l')
        processor = YoloxProcessor('yolox_l')
        tensor = processor(test_images)
        output = model(tensor)
        results = processor.postprocess(test_images, output)
        print(results)
        assert results == [{'bboxes': [(266.55224609375, -0.20783233642578125, 639.8648681640625, 223.93484497070312), (258.4273681640625, 153.58741760253906, 295.0760498046875, 232.3666229248047), (2.655975341796875, 118.57349395751953, 459.0986633300781, 311.5780029296875), (209.98736572265625, 110.51018524169922, 278.888916015625, 140.30113220214844)], 'scores': [0.9547764066322912, 0.9331239584181183, 0.9127988854742739, 0.8015047842898539], 'labels': [7, 12, 2, 2]}, {'bboxes': [(149.04432678222656, 124.79025268554688, 474.92474365234375, 628.5472412109375), (237.06727600097656, 93.3296127319336, 262.45574951171875, 123.57549285888672), (34.858360290527344, 254.82289123535156, 247.63385009765625, 332.03851318359375), (0.06286239624023438, 180.95333862304688, 43.45553970336914, 514.7296142578125)], 'scores': [0.9551582076262335, 0.8987103117490705, 0.8561213796568978, 0.8463778698491708], 'labels': [0, 32, 34, 0]}, {'bboxes': [(8.01373291015625, 192.7410888671875, 619.24365234375, 475.85382080078125), (310.7430419921875, 0.5427398681640625, 629.7476196289062, 244.99168395996094), (1.36309814453125, 14.578079223632812, 433.4184265136719, 370.49749755859375), (258.5732421875, 232.23861694335938, 568.9124145507812, 473.8310852050781)], 'scores': [0.9523050819272143, 0.9383309032892484, 0.9285364752896044, 0.6830796094431122], 'labels': [45, 45, 45, 50]}]
        assert False
