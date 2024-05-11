from PIL import Image
from wmdetection.models import get_watermarks_detection_model
from wmdetection.pipelines.predictor import WatermarksPredictor

# checkpoint is automatically downloaded
model, transforms = get_watermarks_detection_model(
    "convnext-tiny", device="cpu", fp16=False, cache_dir="./weights"
)
predictor = WatermarksPredictor(model, transforms, "cpu")

result = predictor.predict_image(Image.open("images/watermark/3.jpg"))
print("watermarked" if result else "clean")
