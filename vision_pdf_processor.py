"""
Vision-based PDF Processor
Converts PDF pages to images and extracts both text and visual information
Handles large PDFs (up to 500+ pages) efficiently
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any
import io
import base64
from PIL import Image
import fitz  # PyMuPDF for better image handling
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VisionPDFProcessor:
    """
    Advanced PDF processor that treats each page as an image
    Extracts text, images, and visual layout for vision models
    """

    def __init__(
        self,
        dpi: int = 150,  # DPI for page rendering (150 is good balance)
        extract_images: bool = True,
        extract_text: bool = True,
        batch_size: int = 10  # Process pages in batches
    ):
        """
        Initialize Vision PDF Processor.

        Args:
            dpi: Resolution for rendering PDF pages as images
            extract_images: Extract embedded images separately
            extract_text: Extract text content as well
            batch_size: Number of pages to process in each batch
        """
        self.dpi = dpi
        self.extract_images = extract_images
        self.extract_text = extract_text
        self.batch_size = batch_size

    def process_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process entire PDF: convert pages to images, extract text and embedded images.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing:
                - page_images: List of paths to rendered page images
                - page_text: List of extracted text per page
                - embedded_images: List of extracted images with metadata
                - metadata: PDF metadata
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return None

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Open PDF with PyMuPDF
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)

            logger.info(f"Processing PDF with {total_pages} pages...")

            result = {
                'page_images': [],
                'page_text': [],
                'embedded_images': [],
                'metadata': {
                    'total_pages': total_pages,
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'producer': doc.metadata.get('producer', ''),
                }
            }

            # Process pages with progress bar
            for page_num in tqdm(range(total_pages), desc="Processing PDF pages"):
                page = doc[page_num]

                # Convert page to image
                if True:  # Always render pages
                    page_img_path = self._render_page_to_image(
                        page, page_num, output_path
                    )
                    result['page_images'].append(page_img_path)

                # Extract text
                if self.extract_text:
                    text = page.get_text("text")
                    result['page_text'].append({
                        'page': page_num + 1,
                        'text': text.strip()
                    })

                # Extract embedded images
                if self.extract_images:
                    embedded_imgs = self._extract_page_images(
                        page, page_num, output_path
                    )
                    result['embedded_images'].extend(embedded_imgs)

            doc.close()

            logger.info(f"Successfully processed {total_pages} pages")
            logger.info(f"Extracted {len(result['page_images'])} page images")
            logger.info(f"Extracted {len(result['embedded_images'])} embedded images")

            return result

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return None

    def _render_page_to_image(
        self,
        page: fitz.Page,
        page_num: int,
        output_dir: Path
    ) -> str:
        """
        Render a PDF page to an image.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            output_dir: Output directory

        Returns:
            Path to saved image
        """
        try:
            # Calculate zoom for desired DPI (72 DPI is default)
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)

            # Render page to pixmap
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Save as PNG
            img_filename = f"page_{page_num + 1:04d}.png"
            img_path = output_dir / img_filename
            pix.save(str(img_path))

            return str(img_path)

        except Exception as e:
            logger.error(f"Error rendering page {page_num + 1}: {str(e)}")
            return None

    def _extract_page_images(
        self,
        page: fitz.Page,
        page_num: int,
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Extract embedded images from a page.

        Args:
            page: PyMuPDF page object
            page_num: Page number (0-indexed)
            output_dir: Output directory

        Returns:
            List of image metadata dictionaries
        """
        images_info = []

        try:
            # Create subdirectory for embedded images
            embedded_dir = output_dir / "embedded_images"
            embedded_dir.mkdir(exist_ok=True)

            # Get image list from page
            image_list = page.get_images(full=True)

            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]  # Image xref number
                    base_image = page.parent.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Save image
                        img_filename = f"page_{page_num + 1:04d}_img_{img_idx + 1:03d}.{image_ext}"
                        img_path = embedded_dir / img_filename

                        with open(img_path, 'wb') as f:
                            f.write(image_bytes)

                        images_info.append({
                            'page': page_num + 1,
                            'filename': img_filename,
                            'path': str(img_path),
                            'format': image_ext,
                            'xref': xref
                        })

                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_num + 1}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error extracting images from page {page_num + 1}: {str(e)}")

        return images_info

    def image_to_base64(self, image_path: str) -> Optional[str]:
        """
        Convert image to base64 string for sending to vision models.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded string
        """
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            return base64.b64encode(image_bytes).decode('utf-8')

        except Exception as e:
            logger.error(f"Error encoding image to base64: {str(e)}")
            return None

    def get_page_summary(self, page_image_path: str, page_text: str) -> Dict[str, Any]:
        """
        Create a summary of page content combining image and text.

        Args:
            page_image_path: Path to page image
            page_text: Extracted text from page

        Returns:
            Dictionary with page summary information
        """
        return {
            'image_path': page_image_path,
            'text': page_text,
            'image_base64': self.image_to_base64(page_image_path),
            'has_text': len(page_text.strip()) > 0,
            'text_length': len(page_text)
        }


if __name__ == "__main__":
    # Test the processor
    logging.basicConfig(level=logging.INFO)

    processor = VisionPDFProcessor(dpi=150)

    # Test with a sample PDF
    test_pdf = "test.pdf"
    if Path(test_pdf).exists():
        result = processor.process_pdf(test_pdf, "test_output")
        if result:
            print(f"\nProcessed {result['metadata']['total_pages']} pages")
            print(f"Page images: {len(result['page_images'])}")
            print(f"Embedded images: {len(result['embedded_images'])}")
    else:
        print(f"Test PDF not found: {test_pdf}")
