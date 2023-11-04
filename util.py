import supervision as sv
from ultralytics import YOLO
from numpy import ndarray
from cv2 import VideoCapture, imencode, imread
from PIL import Image
import io
from typing import Union
from io import BytesIO


def load_model():
    m = YOLO("./detector.pt")
    m.fuse()
    return m


model = load_model()


def bytes_to_image(binary_image: bytes) -> Image.Image:
    """Convert image from bytes to PIL RGB format

    Args:
        binary_image (bytes): The binary representation of the image

    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image


def image_to_bytes(image: Union[Image.Image, str]) -> BytesIO:
    """
    Convert PIL image to Bytes

    Args:
    image (Image): A PIL image instance

    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    # if image is str type
    image = Image.open(image)

    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=100)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image


def get_results(frame):
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 7]

    return detections


def detect_from_image(image):
    try:
        image = imread(image)
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.class_id == 7]

    except Exception as e:
        raise ValueError("Error detection from image file 0:", e)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [
        results.names[class_id]
        for class_id
        in detections.class_id
    ]

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels)

    return annotated_image, len(detections)


class TruckDetector:
    def __init__(self, source: Union[str, int] = None):
        self.line_zone = None
        self.CLASS_NAMES_DICT = model.model.names
        self.source = source
        self.generator = sv.get_video_frames_generator(source) if source else None
        self.box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        self.trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
        self.line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=2, custom_in_text="Camions entrants", custom_out_text="Camions sortants")
        self.byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    def get_frame(self):
        iterator = iter(self.generator)
        frame = next(iterator)

        return frame

    def annotate_frame(self, frame: ndarray, detections: sv.Detections):
        labels = [
            f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections
        ]

        annotated_frame = self.trace_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        return annotated_frame

    def init_video_counting(self):
        video_info = sv.VideoInfo.from_video_path(self.source)
        LINE_START = sv.Point(0, video_info.height * 0.7)
        LINE_END = sv.Point(video_info.width, video_info.height * 0.7)

        # create LineZone instance
        self.line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

        TARGET_VIDEO_PATH = f"videos/detected_video.mp4"

    def callback(self, frame: ndarray):

        detections = get_results(frame)

        detections = self.byte_tracker.update_with_detections(detections)

        annotated_frame = self.annotate_frame(frame, detections)

        self.line_zone.trigger(detections)

        return self.line_zone_annotator.annotate(annotated_frame, line_counter=self.line_zone)

    def __call__(self, *args, **kwargs):

        if str(self.source).endswith((".jpg", ".png", ".jpeg")):
            try:
                annotated_frame, nb_trucks = detect_from_image(self.source)
            except Exception as e:
                raise ValueError("Error detection from image file:", e)

            _, buffer = imencode(".jpg", annotated_frame)

            IMAGE_HEADER = (b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n")

            annotated_image = IMAGE_HEADER + buffer.tobytes() + b"\r\n"

            return annotated_image, nb_trucks
        else:

            cap = VideoCapture(self.source)
            if not cap.isOpened():
                raise ValueError("Error opening video source")

            while True:

                try:
                    ret, frame = cap.read()
                    if not ret:
                        break
                except Exception as e:
                    print("Error reading video frame:", e)
                    raise e

                annotated_frame = self.callback(frame)

                in_truck = self.line_zone.in_count
                out_truck = self.line_zone.out_count

                nb_trucks = in_truck + out_truck

                return nb_trucks
