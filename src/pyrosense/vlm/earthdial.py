"""
PyroSense EarthDial VLM Integration
====================================

EarthDial Vision-Language Model wrapper for fire area analysis.

Model: akshaydudhane/EarthDial_4B_RGB
- 4.15B parameters (InternVL2: InternViT-300M + Phi-3-mini)
- Supports: classification, detection, captioning, Q&A, visual reasoning
- Input: RGB satellite imagery + natural language query
- Output: Natural language response

Requires: EarthDial package from https://github.com/hiyamdebary/EarthDial
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from pyrosense.vlm.image_utils import ensure_rgb_image
from pyrosense.vlm.prompts import FIRE_ANALYSIS_PROMPT, REPORT_PROMPTS

# Mock decord to avoid video dependency (we only need image processing)
if "decord" not in sys.modules:
    class _MockDecord:
        class VideoReader:
            pass
        def __getattr__(self, name):
            return self.VideoReader
    sys.modules["decord"] = _MockDecord()

if TYPE_CHECKING:
    import torch
    from pyrosense.data import FireEvent

# EarthDial package path - configurable via environment variable
# Default: /tmp/EarthDial/src (clone from https://github.com/hiyamdebary/EarthDial)
EARTHDIAL_SRC_PATH = Path(os.environ.get("EARTHDIAL_SRC_PATH", "/tmp/EarthDial/src"))


def _setup_earthdial_path():
    """Add EarthDial source to path if available."""
    if EARTHDIAL_SRC_PATH.exists():
        src_str = str(EARTHDIAL_SRC_PATH)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
            logger.debug(f"Added EarthDial to path: {src_str}")
        return True
    return False


@dataclass
class EarthDialAssistant:
    """
    EarthDial VLM wrapper for fire area analysis.

    Args:
        model_name: HuggingFace model name
        device: Device to use ("auto", "cuda", "mps", "cpu")
        load_in_8bit: Use 8-bit quantization for lower memory (CUDA only)
        max_new_tokens: Maximum tokens in response

    Example:
        >>> assistant = EarthDialAssistant(device="auto")
        >>> response = assistant.analyze_image("composite.tif")
        >>> print(response)
    """

    model_name: str = "akshaydudhane/EarthDial_4B_RGB"
    device: str = "auto"
    load_in_8bit: bool = False
    max_new_tokens: int = 512

    # Internal state (not configurable)
    _model: object = field(default=None, repr=False)
    _tokenizer: object = field(default=None, repr=False)
    _transform: object = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)
    _loaded: bool = field(default=False, repr=False)

    def __post_init__(self):
        """Detect device on initialization."""
        self._device = self._detect_device()
        logger.info(f"EarthDial initialized (device={self._device}, load_in_8bit={self.load_in_8bit})")

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        if self.device != "auto":
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self) -> bool:
        """
        Load the EarthDial model (lazy loading).

        Returns:
            True if model loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        # Ensure EarthDial package is available
        if not _setup_earthdial_path():
            logger.error(
                "EarthDial package not found. Install with:\n"
                "  cd /tmp && git clone https://github.com/hiyamdebary/EarthDial.git\n"
                "Or set EARTHDIAL_SRC_PATH environment variable to your EarthDial/src path"
            )
            return False

        try:
            # Suppress verbose warnings
            import warnings
            import logging as std_logging
            import os
            warnings.filterwarnings("ignore", message=".*flash.*", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*timm.models.layers.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*Special tokens.*", category=UserWarning)
            warnings.filterwarnings("ignore")  # Suppress all warnings
            # Suppress transformers logging completely
            std_logging.getLogger("transformers").setLevel(std_logging.ERROR)
            std_logging.getLogger("transformers.tokenization_utils_base").setLevel(std_logging.ERROR)
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

            import torch
            from transformers import AutoTokenizer
            from earthdial.model.internvl_chat import InternVLChatModel
            from earthdial.train.dataset import build_transform

            logger.info(f"Loading EarthDial model: {self.model_name}")
            logger.info(f"Device: {self._device}, 8-bit: {self.load_in_8bit}")

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=False,
            )

            # Determine dtype based on device
            if self._device == "cuda":
                dtype = torch.bfloat16
            else:
                # MPS works better with float16
                dtype = torch.float16

            # Load model
            # Note: low_cpu_mem_usage=False because model init uses .item() on tensors
            model_kwargs = {
                "low_cpu_mem_usage": False,
                "torch_dtype": dtype,
            }

            if self.load_in_8bit and self._device == "cuda":
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            elif self._device == "cuda":
                model_kwargs["device_map"] = "auto"

            self._model = InternVLChatModel.from_pretrained(
                self.model_name,
                **model_kwargs,
            ).eval()

            # Move to device if not using device_map
            if "device_map" not in model_kwargs and self._device != "cpu":
                self._model = self._model.to(self._device)

            # Build image transform
            image_size = self._model.config.force_image_size or self._model.config.vision_config.image_size
            self._transform = build_transform(is_train=False, input_size=image_size, normalize_type='imagenet')

            self._loaded = True
            logger.info("EarthDial model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load EarthDial model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _ensure_loaded(self) -> None:
        """Ensure model is loaded before inference."""
        if not self._loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load EarthDial model")

    def _prepare_image(
        self,
        image_path: str | Path,
        center_lat: float | None = None,
        center_lon: float | None = None,
    ):
        """
        Prepare image for model input.

        Crops to 224×224 pixels (6.7 km × 6.7 km) centered on the event location
        to match the area used by Prithvi for fire probability prediction.

        Args:
            image_path: Path to HLS composite or RGB image
            center_lat: Latitude to center crop on (default: center of image)
            center_lon: Longitude to center crop on (default: center of image)
        """
        import torch
        from PIL import Image

        # Ensure RGB format with crop to match Prithvi area
        rgb_path = ensure_rgb_image(
            image_path,
            center_lat=center_lat,
            center_lon=center_lon,
            crop_size=224,
        )

        # Load and transform image
        image = Image.open(rgb_path).convert("RGB")
        pixel_values = self._transform(image).unsqueeze(0)

        # Move to device and dtype
        dtype = torch.bfloat16 if self._device == "cuda" else torch.float16
        if self._device != "cpu":
            pixel_values = pixel_values.to(self._device).to(dtype)
        else:
            pixel_values = pixel_values.to(dtype)

        return pixel_values

    def analyze_image(
        self,
        image_path: str | Path,
        question: str = "Describe this satellite image and assess fire risk factors.",
        center_lat: float | None = None,
        center_lon: float | None = None,
    ) -> str:
        """
        Analyze a satellite image with a single question.

        Args:
            image_path: Path to image (HLS composite or RGB)
            question: Question to ask about the image
            center_lat: Latitude to center crop on (default: center of image)
            center_lon: Longitude to center crop on (default: center of image)

        Returns:
            Model's response
        """
        self._ensure_loaded()

        # Prepare image (crops to 6.7 km × 6.7 km area)
        pixel_values = self._prepare_image(image_path, center_lat, center_lon)

        # Generate response
        try:
            response = self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config={
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                },
                verbose=False,
            )
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def chat(
        self,
        image_path: str | Path,
        question: str,
        history: list | None = None,
        center_lat: float | None = None,
        center_lon: float | None = None,
    ) -> tuple[str, list]:
        """
        Multi-turn conversation with image context.

        Args:
            image_path: Path to image
            question: Current question
            history: Previous conversation history
            center_lat: Latitude to center crop on (default: center of image)
            center_lon: Longitude to center crop on (default: center of image)

        Returns:
            Tuple of (response, updated_history)
        """
        self._ensure_loaded()

        # Prepare image (crops to 6.7 km × 6.7 km area)
        pixel_values = self._prepare_image(image_path, center_lat, center_lon)

        # Generate response with history
        # NOTE: Pass history as-is (including None) - the model handles None internally
        # and adds <image> token to the question only when history is None
        try:
            response, new_history = self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                history=history,
                generation_config={
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                },
                return_history=True,
                verbose=False,
            )
            return response, new_history
        except Exception as e:
            logger.error(f"Chat failed: {type(e).__name__}: {e}")
            raise

    def generate_report(
        self,
        image_path: str | Path,
        event: FireEvent | None = None,
        fire_probability: float | None = None,
        center_lat: float | None = None,
        center_lon: float | None = None,
    ) -> dict[str, str]:
        """
        Generate a structured fire risk report.

        Args:
            image_path: Path to satellite image
            event: Optional FireEvent with location/date info
            fire_probability: Optional model-predicted fire probability
            center_lat: Latitude to center crop on (uses event.latitude if not provided)
            center_lon: Longitude to center crop on (uses event.longitude if not provided)

        Returns:
            Dictionary with report sections:
            - summary: Brief overview
            - vegetation_analysis: Vegetation description
            - terrain_factors: Terrain analysis
            - recommended_strategies: Firefighting strategies
            - risk_assessment: Risk level and justification
        """
        self._ensure_loaded()

        # Use event coordinates if available
        if event is not None and center_lat is None:
            center_lat = event.latitude
            center_lon = event.longitude

        # Prepare image once (crops to 6.7 km × 6.7 km area)
        pixel_values = self._prepare_image(image_path, center_lat, center_lon)

        report = {}

        # Generate summary
        summary_prompt = REPORT_PROMPTS["summary"]
        if event and fire_probability is not None:
            summary_prompt = f"Fire probability: {fire_probability:.1%}. " + summary_prompt
        report["summary"] = self._generate(pixel_values, summary_prompt)

        # Generate each section
        report["vegetation_analysis"] = self._generate(pixel_values, REPORT_PROMPTS["vegetation"])
        report["terrain_factors"] = self._generate(pixel_values, REPORT_PROMPTS["terrain"])
        report["recommended_strategies"] = self._generate(pixel_values, REPORT_PROMPTS["strategies"])
        report["risk_assessment"] = self._generate(pixel_values, REPORT_PROMPTS["risk"])

        return report

    def _generate(self, pixel_values, prompt: str) -> str:
        """Internal helper to generate response for prepared image and prompt."""
        try:
            response = self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=prompt,
                generation_config={
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": False,
                    "num_beams": 1,
                },
                verbose=False,
            )
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"[Error: {e}]"

    def analyze_with_context(
        self,
        image_path: str | Path,
        event: FireEvent,
        fire_probability: float,
    ) -> str:
        """
        Analyze image with full fire event context.

        Crops to 6.7 km × 6.7 km area centered on the event location
        to match the area used by Prithvi for fire probability prediction.

        Args:
            image_path: Path to satellite image
            event: FireEvent with location and date
            fire_probability: Model-predicted probability

        Returns:
            Detailed analysis response
        """
        prompt = FIRE_ANALYSIS_PROMPT.format(
            lat=event.latitude,
            lon=event.longitude,
            date=event.date.strftime("%Y-%m-%d"),
            probability=fire_probability,
        )
        return self.analyze_image(
            image_path,
            prompt,
            center_lat=event.latitude,
            center_lon=event.longitude,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def device_info(self) -> dict:
        """Get device information."""
        import torch

        return {
            "device": self._device,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "load_in_8bit": self.load_in_8bit,
            "model_loaded": self._loaded,
        }
