from __future__ import annotations

"""Docker build helper that preloads Docling model artifacts into the image."""

import os
from pathlib import Path

from docling.utils.model_downloader import download_models


def main() -> None:
    """Download the Docling model bundle during Docker image build."""

    artifacts_dir = Path(os.getenv("DOCLING_ARTIFACTS_PATH", "/opt/docling-models"))
    download_models(
        output_dir=artifacts_dir,
        progress=False,
        with_layout=True,
        with_tableformer=True,
        with_tableformer_v2=False,
        with_code_formula=True,
        with_picture_classifier=True,
        with_smolvlm=False,
        with_granitedocling=False,
        with_granitedocling_mlx=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
        with_granite_chart_extraction=False,
        with_granite_chart_extraction_v4=False,
        with_rapidocr=True,
        with_easyocr=False,
    )


if __name__ == "__main__":
    main()
