"""RT-DETR-v2 comic detector (skeleton / experimental).

Wraps the Apache-2.0 Hugging Face model `ogkalu/comic-text-and-bubble-detector`
(RT-DETR-v2 r50vd) as a pluggable detector. It detects three classes:
    0 = bubble        (speech-balloon region)
    1 = text_bubble   (text inside a bubble)
    2 = text_free     (text outside bubbles, e.g. SFX / narration)

IMPORTANT — read before using in production:
  * This is a BOX detector. It returns rectangles, not stroke-level text masks.
    The erase/inpaint stage in extract.py expects a text mask; here we synthesize
    a *box-filled* mask, which is COARSER than the dbnet/ctd stroke masks and will
    leave/erase more than necessary. For clean inpainting, keep dbnet/ctd as the
    mask source and use this detector for detection/region-typing experiments
    (see detector_ab.py to measure the difference on your own pages first).
  * Requires `transformers` (+ torch). Both are imported lazily so the rest of the
    package still works without them; a clear error is raised only if you select
    this detector without the dependency installed.
  * Training-data provenance of the HF weights is not disclosed on the model card
    (~11k mixed comic images). Verify licensing/data before any commercial use —
    the Apache-2.0 tag covers the code/weights, not necessarily the training data.
"""

import numpy as np
import cv2

from .common import OfflineDetector
from ..utils import Quadrilateral


# Class ids from the model card.
CLS_BUBBLE = 0
CLS_TEXT_BUBBLE = 1
CLS_TEXT_FREE = 2
TEXT_CLASSES = (CLS_TEXT_BUBBLE, CLS_TEXT_FREE)


class RTDetrV2Detector(OfflineDetector):
    # Hugging Face repo id. Override here (or subclass) to point at your own weights.
    _HF_MODEL_ID = 'ogkalu/comic-text-and-bubble-detector'
    # No _MODEL_MAPPING: weights are fetched/cached by transformers `from_pretrained`,
    # so download()/is_downloaded() are effectively no-ops for this detector.
    _MODEL_MAPPING = {}

    async def _load(self, device: str):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as e:
            raise ImportError(
                "RT-DETR detector needs `transformers` (and torch). Install with "
                "`pip install transformers` — or choose another detector "
                "(detector = \"ctd\" / \"default\")."
            ) from e
        self._torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self._HF_MODEL_ID)
        self.model = AutoModelForObjectDetection.from_pretrained(self._HF_MODEL_ID)
        self.model.to(device)
        self.model.eval()
        self.logger.warning(
            "RT-DETR produces BOX-level masks (not stroke masks); inpainting will be "
            "coarser than dbnet/ctd. Recommended for detection/region-typing only."
        )

    async def _unload(self):
        self.model = None
        self.processor = None

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):
        torch = self._torch
        im_h, im_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detection confidence gate. box_threshold (config [detector]) is reused here;
        # RT-DETR usually wants a lower value than dbnet — try ~0.3 in your config.
        conf = float(box_threshold)

        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([[im_h, im_w]], device=self.device)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf
        )[0]

        scores = result["scores"].detach().cpu().numpy()
        labels = result["labels"].detach().cpu().numpy().astype(int)
        boxes = result["boxes"].detach().cpu().numpy()  # xyxy in original-image pixels

        textlines = []
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
            if label not in TEXT_CLASSES:
                continue  # bubble (class 0) is grouping context, not a text region
            x1 = int(max(0, min(im_w, x1))); x2 = int(max(0, min(im_w, x2)))
            y1 = int(max(0, min(im_h, y1))); y2 = int(max(0, min(im_h, y2)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            textlines.append(Quadrilateral(pts, '', float(score)))
            # Box-filled mask (see module docstring caveat).
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

        if verbose:
            self.logger.info(f"RT-DETR: {len(textlines)} text region(s) "
                             f"(of {len(boxes)} detections) at conf>={conf}")

        # (textlines, raw_mask, refined_mask) — refined left None, like ctd.
        return textlines, mask, None
