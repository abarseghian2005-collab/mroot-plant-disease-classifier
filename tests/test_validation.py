import copy
import io
import unittest
from PIL import Image
from src.mroot.config import load_config, validate_config
from src.mroot.validation import InvalidImage, decode_image, validate_manifest, validate_training_splits


class InputValidationTests(unittest.TestCase):
    def test_rgb_conversion(self):
        stream = io.BytesIO()
        Image.new("L", (16, 12), color=120).save(stream, format="PNG")
        image = decode_image(stream.getvalue())
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (16, 12))

    def test_empty_and_invalid_files(self):
        for data in [b"", b"this is not a photo"]:
            with self.subTest(data=data):
                with self.assertRaises(InvalidImage):
                    decode_image(data)

    def test_unsupported_format(self):
        stream = io.BytesIO()
        Image.new("RGB", (8, 8)).save(stream, format="GIF")
        with self.assertRaises(InvalidImage):
            decode_image(stream.getvalue())

    def test_frozen_configuration(self):
        self.assertEqual(len(load_config()["classes"]), 39)
        for mutate in [
            lambda c: c["classes"].pop(),
            lambda c: c["members"][0].update(weight=0.7),
            lambda c: c["members"][0].update(weights_file="../outside.safetensors"),
            lambda c: c["members"][0].update(sha256="wrong"),
            lambda c: c["views"].update(normal=0.5),
        ]:
            config = copy.deepcopy(load_config())
            mutate(config)
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_committed_training_splits(self):
        from src.mroot.config import PROJECT_ROOT
        mapping = load_config()["class_to_idx"]
        train = validate_manifest(PROJECT_ROOT / "data/manifests/experiment_train.csv", mapping, 6790)
        val = validate_manifest(PROJECT_ROOT / "data/manifests/experiment_validation.csv", mapping, 388)
        validate_training_splits(train, val)

    def test_detects_split_overlap(self):
        from src.mroot.validation import ManifestRow
        row = ManifestRow(path="same.jpg", disease="rust", label=30, duplicate_group="group1")
        with self.assertRaises(ValueError):
            validate_training_splits([row], [row])


if __name__ == "__main__":
    unittest.main()
