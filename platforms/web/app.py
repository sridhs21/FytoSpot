import os
import numpy as np
import cv2
import base64
import json
import logging
import traceback
from io import BytesIO
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    send_from_directory,
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)


UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def load_species_names(file_path):
    try:
        with open(file_path, "r") as f:
            species_names = json.load(f)
        return species_names
    except Exception as e:
        logger.error(f"Error loading species names: {e}")
        return {}


def format_species_name(raw_name, species_names=None):

    if isinstance(raw_name, str) and not raw_name.isdigit():

        name = raw_name.replace("_", " ")
        return name.title()

    if species_names and str(raw_name) in species_names:
        raw_name = species_names[str(raw_name)]
        return raw_name.replace("_", " ").title()

    return f"Plant {raw_name}"


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


try:

    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.extend([current_dir, parent_dir, os.path.dirname(parent_dir)])

    from core.detection.plant_detector import PlantDetector
    from core.models.plant_identifier import PlantIdentifier
    from core.data.knowledge_base import prepare_knowledge_base
    from core.models.wrapper import TrainedModelWrapper

    logger.info("Successfully imported core modules")

    plant_detector = PlantDetector()
    logger.info("Initialized PlantDetector")
except Exception as e:
    logger.error(f"Error importing core modules: {e}")
    logger.error(traceback.format_exc())

    class MockPlantDetector:
        def __init__(self):
            self.detection_method = "multi"

        def set_detection_method(self, method):
            self.detection_method = method
            logger.info(f"Set detection method to {method}")

        def detect_object(self, frame):

            h, w = frame.shape[:2]
            bbox = (int(w * 0.25), int(h * 0.25), int(w * 0.5), int(h * 0.5))

            debug_frame = frame.copy()
            x, y, w, h = bbox
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            return bbox, debug_frame

    plant_detector = MockPlantDetector()
    logger.info("Using MockPlantDetector for testing")


if not hasattr(plant_detector, "set_detection_method"):
    logger.info("Patching PlantDetector with set_detection_method")

    def set_detection_method(self, method):
        logger.info(f"Setting detection method to: {method}")

        self.detection_method = method

        method_mapping = {
            "grabcut": "multi",
            "motion": "multi",
            "yolo": "contour",
        }

        if method in method_mapping:
            mapped_method = method_mapping[method]
            logger.info(f"Method '{method}' mapped to '{mapped_method}'")
            self.detection_method = mapped_method

    setattr(plant_detector.__class__, "set_detection_method", set_detection_method)

    setattr(plant_detector, "detection_method", "multi")


if not hasattr(plant_detector, "detect_object"):
    logger.info("Patching PlantDetector with detect_object method")

    def detect_object(self, frame):

        logger.info(
            f"Using detect_object with method: {getattr(self, 'detection_method', 'multi')}"
        )

        if hasattr(self, "detect"):

            bbox, confidence = self.detect(frame)

            if hasattr(self, "visualize"):
                display_frame, debug_frame = self.visualize(frame, bbox, confidence)
            else:

                debug_frame = frame.copy()
                if bbox is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        debug_frame,
                        f"Confidence: {confidence:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            return bbox, debug_frame
        else:

            logger.warning("No detect method found, falling back to basic detection")

            try:

                detection_method = getattr(self, "detection_method", "multi")

                if detection_method == "color" and hasattr(self, "color_filter"):

                    mask = self.color_filter(frame)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    min_area = 1000
                    valid_contours = [
                        cnt for cnt in contours if cv2.contourArea(cnt) > min_area
                    ]

                    bbox = None
                    if valid_contours:

                        largest_contour = max(valid_contours, key=cv2.contourArea)
                        bbox = cv2.boundingRect(largest_contour)

                    debug_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                elif detection_method == "texture" and hasattr(self, "texture_filter"):

                    mask = self.texture_filter(frame)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    min_area = 1000
                    valid_contours = [
                        cnt for cnt in contours if cv2.contourArea(cnt) > min_area
                    ]

                    bbox = None
                    if valid_contours:

                        largest_contour = max(valid_contours, key=cv2.contourArea)
                        bbox = cv2.boundingRect(largest_contour)

                    debug_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                elif detection_method == "contour" and hasattr(
                    self, "contour_analysis"
                ):

                    mask = self.contour_analysis(frame)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    min_area = 1000
                    valid_contours = [
                        cnt for cnt in contours if cv2.contourArea(cnt) > min_area
                    ]

                    bbox = None
                    if valid_contours:

                        largest_contour = max(valid_contours, key=cv2.contourArea)
                        bbox = cv2.boundingRect(largest_contour)

                    debug_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                else:

                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    lower_green = np.array([35, 40, 40])
                    upper_green = np.array([85, 255, 255])

                    mask = cv2.inRange(hsv, lower_green, upper_green)

                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    min_area = 1000
                    valid_contours = [
                        cnt for cnt in contours if cv2.contourArea(cnt) > min_area
                    ]

                    bbox = None
                    if valid_contours:

                        largest_contour = max(valid_contours, key=cv2.contourArea)
                        bbox = cv2.boundingRect(largest_contour)

                    debug_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                if bbox is not None:
                    x, y, w, h = bbox

                    display_frame = frame.copy()
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        "Plant Detected",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        debug_frame,
                        f"Method: {detection_method}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                return bbox, debug_frame

            except Exception as e:
                logger.error(f"Error in fallback detection: {str(e)}")
                logger.error(traceback.format_exc())

                h, w = frame.shape[:2]
                bbox = (
                    int(w * 0.25),
                    int(h * 0.25),
                    int(w * 0.5),
                    int(h * 0.5),
                )

                debug_frame = frame.copy()
                x, y, w, h = bbox
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    debug_frame,
                    "Plant (Fallback)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                return bbox, debug_frame

    setattr(plant_detector.__class__, "detect_object", detect_object)


plant_identifier = None
try:

    ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    model_path = os.path.join(ROOT_DIR, "models/trained/final_model.pt")
    class_mapping_path = os.path.join(ROOT_DIR, "models/configs/class_mapping.json")
    knowledge_base_path = os.path.join(ROOT_DIR, "models/knowledge_base.json")
    species_names_path = os.path.join(
        ROOT_DIR, "data/plantnet_300K/plantnet300K_species_names.json"
    )

    species_names = load_species_names(species_names_path)
    logger.info(f"Loaded {len(species_names)} species names")

    if "prepare_knowledge_base" in globals() and not os.path.exists(
        knowledge_base_path
    ):
        knowledge_base_path = prepare_knowledge_base(knowledge_base_path)

    if os.path.exists(model_path) and os.path.exists(class_mapping_path):
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)

        model_wrapper = TrainedModelWrapper(model_path, len(class_mapping), device)

        plant_identifier = PlantIdentifier(
            model_path=None,
            class_mapping_path=class_mapping_path,
            knowledge_base_path=knowledge_base_path,
            confidence_threshold=0.5,
        )

        plant_identifier.model = model_wrapper

        logger.info("Plant identifier initialized successfully")
    else:
        logger.warning(f"Model files not found. Looking in: {model_path}")
        logger.warning("Plant identification will not be available.")
except Exception as e:
    logger.error(f"Error initializing plant identifier: {e}")
    logger.error(traceback.format_exc())
    plant_identifier = None
    species_names = {}


if plant_identifier is None:

    class MockPlantIdentifier:
        def __init__(self):
            pass

        def identify_plant(self, image):

            return {
                "status": "success",
                "message": "Plant identified (DEMO MODE)",
                "class_id": 2,
                "confidence": 0.85,
                "class_name": "Geranium",
                "top_predictions": [
                    {
                        "class_id": 2,
                        "class_name": "Geranium",
                        "probability": 0.85,
                    },
                    {
                        "class_id": 3,
                        "class_name": "Petunia",
                        "probability": 0.10,
                    },
                    {
                        "class_id": 5,
                        "class_name": "Begonia",
                        "probability": 0.03,
                    },
                ],
                "plant_info": {
                    "scientific_name": "Pelargonium x hortorum",
                    "family": "Geraniaceae",
                    "description": "Geraniums are popular flowering plants known for their colorful blooms and aromatic foliage.",
                    "care": {
                        "watering": "Water when soil is dry to touch",
                        "sunlight": "Full to partial sun",
                        "soil": "Well-draining soil",
                        "temperature": "Prefer moderate temperatures",
                    },
                    "common_varieties": [
                        "Zonal Geranium",
                        "Ivy Geranium",
                        "Scented Geranium",
                    ],
                },
            }

    plant_identifier = MockPlantIdentifier()
    logger.info("Using MockPlantIdentifier for testing")


@app.before_request
def before_request():
    if request.path == "/detect":
        logger.info(f"Received request to {request.path}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Form data: {request.form}")
        logger.info(f"Files: {request.files}")


for directory in [app.config["UPLOAD_FOLDER"]]:
    os.makedirs(directory, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


@app.route("/detect", methods=["POST"])
def detect():
    try:
        logger.info("Detect endpoint called")
        if "image" not in request.files:
            logger.warning("No image in request")
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            logger.warning("Empty filename")
            return jsonify({"error": "No image selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        logger.info(f"Image saved to {filepath}")

        frame = cv2.imread(filepath)
        if frame is None:
            logger.error(f"Could not read image from {filepath}")
            return jsonify({"error": "Could not read image"}), 400

        detection_method = request.form.get("detection_method", "multi")
        logger.info(f"Using detection method: {detection_method}")

        if hasattr(plant_detector, "set_detection_method"):
            plant_detector.set_detection_method(detection_method)

        try:

            import inspect

            sig = inspect.signature(plant_detector.detect_object)

            if len(sig.parameters) == 1:

                bbox, debug_frame = plant_detector.detect_object(frame)
            else:

                if hasattr(plant_detector, "detect"):
                    logger.info("Falling back to detect method")
                    bbox, confidence = plant_detector.detect(frame)

                    if hasattr(plant_detector, "visualize"):
                        _, debug_frame = plant_detector.visualize(
                            frame, bbox, confidence
                        )
                    else:

                        debug_frame = frame.copy()
                        if bbox:
                            x, y, w, h = bbox
                            cv2.rectangle(
                                debug_frame,
                                (x, y),
                                (x + w, y + h),
                                (0, 255, 0),
                                2,
                            )
                else:
                    logger.error("No suitable detection method found")
                    return (
                        jsonify({"error": "No suitable detection method available"}),
                        500,
                    )

            logger.info(f"Detection result: bbox={bbox}")
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Detection error: {str(e)}"}), 500

        response = {
            "detection": bbox is not None,
            "bbox": bbox if bbox is not None else None,
        }

        identification_results = None

        if bbox is not None:
            x, y, w, h = bbox

            if (
                x >= 0
                and y >= 0
                and w > 0
                and h > 0
                and x + w <= frame.shape[1]
                and y + h <= frame.shape[0]
            ):

                crop = frame[y : y + h, x : x + w]

                crop_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], f"crop_{filename}"
                )
                cv2.imwrite(crop_path, crop)
                session["crop_path"] = crop_path
                logger.info(f"Crop saved to {crop_path}")

                if plant_identifier is not None:
                    try:

                        logger.info("Identifying plant immediately after detection")
                        identification = plant_identifier.identify_plant(crop)

                        if "class_name" in identification:
                            class_id = identification.get("class_id")
                            class_name = identification.get("class_name")
                            formatted_name = format_species_name(
                                class_name or class_id, species_names
                            )
                            identification["class_name"] = formatted_name

                        if "top_predictions" in identification:
                            for pred in identification["top_predictions"]:
                                if "class_name" in pred:
                                    class_id = pred.get("class_id")
                                    class_name = pred.get("class_name")
                                    formatted_name = format_species_name(
                                        class_name or class_id, species_names
                                    )
                                    pred["class_name"] = formatted_name

                        identification_results = identification
                        response["identification"] = identification
                        logger.info(
                            f"Automatic identification complete: {identification.get('status')}"
                        )
                    except Exception as e:
                        logger.error(f"Error in automatic identification: {str(e)}")
                        logger.error(traceback.format_exc())

                        response["identification_error"] = str(e)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        if debug_frame is not None:
            _, debug_buffer = cv2.imencode(".jpg", debug_frame)
            debug_b64 = base64.b64encode(debug_buffer).decode("utf-8")
        else:
            debug_b64 = None

        response["frame"] = frame_b64
        response["debug_frame"] = debug_b64

        logger.info("Detection complete, sending response")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in detect endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Detection error: {str(e)}"}), 500


@app.route("/identify", methods=["POST"])
def identify():
    try:
        logger.info("Identify endpoint called")
        if plant_identifier is None:
            logger.warning("Plant identifier not available")
            return (
                jsonify({"error": "Plant identification model not available"}),
                400,
            )

        crop_path = session.get("crop_path")
        if not crop_path or not os.path.exists(crop_path):
            logger.warning(
                f"No crop path found in session or file doesn't exist: {crop_path}"
            )
            return (
                jsonify({"error": "No detected plant region available"}),
                400,
            )

        crop = cv2.imread(crop_path)
        if crop is None:
            logger.error(f"Could not read cropped image from {crop_path}")
            return jsonify({"error": "Could not read cropped image"}), 400

        logger.info("Identifying plant")
        identification = plant_identifier.identify_plant(crop)
        logger.info(f"Identification result status: {identification.get('status')}")

        if "class_name" in identification:
            class_id = identification.get("class_id")
            class_name = identification.get("class_name")
            formatted_name = format_species_name(class_name or class_id, species_names)
            identification["class_name"] = formatted_name

        if "top_predictions" in identification:
            for pred in identification["top_predictions"]:
                if "class_name" in pred:
                    class_id = pred.get("class_id")
                    class_name = pred.get("class_name")
                    formatted_name = format_species_name(
                        class_name or class_id, species_names
                    )
                    pred["class_name"] = formatted_name

        return jsonify(identification)
    except Exception as e:
        logger.error(f"Error in identify endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error identifying plant: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload_image():
    try:
        logger.info("Upload endpoint called")
        if "image" not in request.files:
            logger.warning("No image in request")
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            logger.warning("Empty filename")
            return jsonify({"error": "No image selected"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        logger.info(f"Image saved to {filepath}")

        session["crop_path"] = filepath

        return jsonify({"success": True, "filename": filename})
    except Exception as e:
        logger.error(f"Error in upload endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Upload error: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    health_status = {
        "status": "up",
        "detector": "initialized",
        "identifier": (
            "initialized" if plant_identifier is not None else "not available"
        ),
    }
    return jsonify(health_status)


@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template("index.html"), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    logger.info("Starting FytóSpot Web Server")
    logger.info(f"Server running on port {port}")
    logger.info(f"Plant detector type: {type(plant_detector).__name__}")
    logger.info(f"Plant identifier available: {plant_identifier is not None}")
    app.run(host="0.0.0.0", port=port, debug=True)
