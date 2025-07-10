import argparse
import glob
import io
import json
import os
import random
import warnings

from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.errors import PdfReadError, PdfReadWarning
from reportlab.lib import colors  # type: ignore
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
from reportlab.lib.units import inch  # type: ignore
from reportlab.pdfbase import pdfmetrics  # type: ignore
from reportlab.pdfbase.ttfonts import TTFont  # type: ignore
from reportlab.pdfgen import canvas  # type: ignore
from reportlab.platypus import Paragraph  # type: ignore
from tqdm import tqdm  # type: ignore


# --- CORRECTED FUNCTION ---
def create_invisible_text_overlay(
    text_to_add, page_width, page_height, font_size, font_name, position="top"
):
    """
    Creates an in-memory PDF page with a block of invisible, wrapping text.
    This version uses the correct low-level PDF command for invisibility.

    Args:
        position: "top" for top of page, "bottom" for bottom of page
    """
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))

    # --- THE FIX ---
    # Save the current graphics state.
    can.saveState()
    # Use a low-level PDF command to set the text render mode to 3 (invisible).
    # This is the correct way to apply this state before drawing a high-level Paragraph.
    can._code.append("3 Tr")
    # --- END OF FIX ---

    # Define the bounding box for our text paragraph (1-inch margins)
    margin = 1 * inch
    frame_width = page_width - 2 * margin
    frame_height = page_height - 2 * margin

    # Create a custom style for our paragraph
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        name="Invisible",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.2,
        textColor=colors.black,
    )

    # More robust text handling to preserve all content
    import html

    # Escape ALL HTML/XML characters to prevent interpretation
    escaped_text = html.escape(text_to_add, quote=True)
    # Replace newlines with <br/> tags AFTER escaping
    text_with_breaks = escaped_text.replace("\n", "<br/>")

    # Create the Paragraph object
    try:
        p = Paragraph(text_with_breaks, style)
    except Exception as e:
        print(f"Error creating Paragraph: {e}")
        # Fallback: use simpler text without any formatting
        simple_text = (
            text_to_add.replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("&", "&amp;")
            .replace("\n", "<br/>")
        )
        p = Paragraph(simple_text, style)

    # Wrap and draw the paragraph from the specified position
    w, h = p.wrap(frame_width, frame_height)

    if position == "bottom":
        # Position at bottom of page
        y_position = margin
    else:
        # Position at top of page (default)
        y_position = page_height - margin - h

    p.drawOn(can, margin, y_position)

    # Restore the canvas state to normal
    can.restoreState()
    can.save()
    packet.seek(0)

    return PdfReader(packet)


# --- END CORRECTED FUNCTION ---


def inject_text_into_pdf(
    input_path, output_path, invisible_text, font_size, injection_locus="first"
):
    """
    Injects invisible text into the PDF at the specified location.

    Args:
        injection_locus: "first" for first page top, "last" for last page bottom
    """
    if not os.path.exists(input_path):
        print(f"Error: Input PDF file not found at '{input_path}'")
        return

    # Use DejaVuSans.ttf from fonts directory
    font_path = os.path.join("data", "fonts", "dejavusans.ttf")
    if not os.path.exists(font_path):
        print(f"Error: DejaVuSans font not found at '{font_path}'")
        return

    font_name_to_use = "DejaVuSans"
    try:
        pdfmetrics.registerFont(TTFont(font_name_to_use, font_path))
        print(f"Successfully registered DejaVuSans font: {font_path}")
    except Exception as e:
        print(f"Error: Failed to register DejaVuSans font. Details: {e}")
        return

    print(f"Reading original PDF: {input_path}")
    existing_pdf = PdfReader(input_path)
    output_writer = PdfWriter()

    if not existing_pdf.pages:
        print("Error: The input PDF appears to be empty or corrupted.")
        return

    # Determine target page based on injection locus
    if injection_locus == "last":
        target_page_index = len(existing_pdf.pages) - 1
        target_page = existing_pdf.pages[target_page_index]
        position = "bottom"
        print(f"Injecting text at bottom of last page ({target_page_index + 1})...")
    else:
        target_page_index = 0
        target_page = existing_pdf.pages[target_page_index]
        position = "top"
        print("Injecting text at top of first page...")

    page_width = target_page.mediabox.width
    page_height = target_page.mediabox.height

    print(
        f"Creating invisible, wrapping text overlay with font '{font_name_to_use}' (size {font_size})..."
    )
    overlay_pdf = create_invisible_text_overlay(
        invisible_text, page_width, page_height, font_size, font_name_to_use, position
    )
    overlay_page = overlay_pdf.pages[0]

    # Handle injection based on locus
    if injection_locus == "last":
        # Copy all pages except the last one
        for i in range(len(existing_pdf.pages) - 1):
            output_writer.add_page(existing_pdf.pages[i])

        # Create modified last page
        print("Injecting text as content at bottom of last page...")
        new_last_page = PageObject.create_blank_page(
            width=page_width, height=page_height
        )
        new_last_page.merge_page(target_page)
        new_last_page.merge_page(overlay_page)
        output_writer.add_page(new_last_page)
    else:
        # First page injection (existing behavior)
        print("Injecting text as the first content element on page 1...")
        new_first_page = PageObject.create_blank_page(
            width=page_width, height=page_height
        )
        new_first_page.merge_page(overlay_page)
        new_first_page.merge_page(target_page)
        output_writer.add_page(new_first_page)

        # Copy remaining pages
        if len(existing_pdf.pages) > 1:
            print(f"Copying the remaining {len(existing_pdf.pages) - 1} page(s)...")
            for i in range(1, len(existing_pdf.pages)):
                output_writer.add_page(existing_pdf.pages[i])

    print(f"Writing new PDF to: {output_path}")
    with open(output_path, "wb") as output_file:
        output_writer.write(output_file)

    print("\nInjection complete!")
    print(
        f"The new file '{output_path}' has been created with text injected at {injection_locus} position."
    )


def read_prompts_json(json_path):
    """
    Reads the prompts.json file and returns the prompt configurations.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_path}'")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_path}': {e}")
        return None
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None


def inject_text_into_pdf_silent(
    input_path,
    output_path,
    invisible_text,
    font_size,
    font_path=None,
    injection_locus="first",
):
    """
    Silent version of inject_text_into_pdf for batch processing (no print statements).
    Uses DejaVuSans.ttf from fonts directory exclusively.

    Args:
        injection_locus: "first" for first page top, "last" for last page bottom
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input PDF file not found at '{input_path}'")

    # Use provided font path or default to DejaVuSans.ttf from fonts directory
    if font_path is None:
        font_path = os.path.join("data", "fonts", "dejavusans.ttf")

    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found at '{font_path}'")

    font_name_to_use = "DejaVuSans"
    try:
        pdfmetrics.registerFont(TTFont(font_name_to_use, font_path))
    except Exception as e:
        raise Exception(f"Failed to register font: {e}")

    # Suppress PyPDF warnings for corrupted PDFs
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PdfReadWarning)
        warnings.filterwarnings("ignore", message=".*wrong pointing object.*")
        warnings.filterwarnings("ignore", message=".*not defined.*")

        try:
            # Try to read PDF with strict=False to handle corrupted files
            existing_pdf = PdfReader(input_path, strict=False)
        except (PdfReadError, Exception) as e:
            raise Exception(f"Failed to read PDF (possibly corrupted): {e}")

    output_writer = PdfWriter()

    if not existing_pdf.pages or len(existing_pdf.pages) == 0:
        raise Exception("The input PDF appears to be empty or corrupted")

    # Determine target page and position based on injection locus
    if injection_locus == "last":
        target_page_index = len(existing_pdf.pages) - 1
        target_page = existing_pdf.pages[target_page_index]
        position = "bottom"
    else:
        target_page_index = 0
        target_page = existing_pdf.pages[target_page_index]
        position = "top"

    try:
        page_width = float(target_page.mediabox.width)
        page_height = float(target_page.mediabox.height)

        # Validate page dimensions
        if page_width <= 0 or page_height <= 0:
            raise Exception("Invalid page dimensions")

    except Exception as e:
        raise Exception(f"Failed to access PDF page properties: {e}")

    try:
        overlay_pdf = create_invisible_text_overlay(
            invisible_text,
            page_width,
            page_height,
            font_size,
            font_name_to_use,
            position,
        )
        overlay_page = overlay_pdf.pages[0]

        # Handle injection based on locus
        if injection_locus == "last":
            # Copy all pages except the last one
            for i in range(len(existing_pdf.pages) - 1):
                try:
                    page = existing_pdf.pages[i]
                    if hasattr(page, "mediabox") and page.mediabox:
                        output_writer.add_page(page)
                except Exception:
                    continue

            # Create modified last page
            new_last_page = PageObject.create_blank_page(
                width=page_width, height=page_height
            )
            new_last_page.merge_page(target_page)
            new_last_page.merge_page(overlay_page)
            output_writer.add_page(new_last_page)
        else:
            # First page injection (existing behavior)
            new_first_page = PageObject.create_blank_page(
                width=page_width, height=page_height
            )
            new_first_page.merge_page(overlay_page)
            new_first_page.merge_page(target_page)
            output_writer.add_page(new_first_page)

            # Copy remaining pages with error handling
            if len(existing_pdf.pages) > 1:
                for i in range(1, len(existing_pdf.pages)):
                    try:
                        page = existing_pdf.pages[i]
                        if hasattr(page, "mediabox") and page.mediabox:
                            output_writer.add_page(page)
                    except Exception:
                        continue
    except Exception as e:
        raise Exception(f"Failed to create or merge overlay: {e}")

    # Copy remaining pages with error handling is now handled above
    # Remove the duplicate page copying code

    try:
        with open(output_path, "wb") as output_file:
            output_writer.write(output_file)
    except Exception as e:
        raise Exception(f"Failed to write output PDF: {e}")


def debug_prompt_text(prompt_text):
    """
    Debug function to verify prompt text integrity
    """
    print(f"Debug: Prompt length: {len(prompt_text)} characters")
    print(f"Debug: First 100 characters: {repr(prompt_text[:100])}")
    print(f"Debug: Last 100 characters: {repr(prompt_text[-100:])}")
    print(f"Debug: Contains newlines: {'\\n' in prompt_text}")
    print(f"Debug: Contains quotes: {'"' in prompt_text}")
    return prompt_text


def process_batch_injection(
    prompts_json_path, pdfs_dir, font_size=1.0, font_path=None, injection_locus="first"
):
    """
    Process batch injection for all PDFs using all prompts from the JSON file.

    Args:
        injection_locus: "first" for first page top, "last" for last page bottom
    """
    # Read the JSON configuration
    print(f"Reading prompts configuration from: {prompts_json_path}")
    prompts_data = read_prompts_json(prompts_json_path)

    if not prompts_data:
        return

    # Find all PDF files in the pdfs directory
    print(f"Scanning for PDF files in: {pdfs_dir}")
    pdf_pattern = os.path.join(pdfs_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        print(f"No PDF files found in '{pdfs_dir}'")
        return

    print(f"Found {len(pdf_files)} PDF files to process")

    # Process each attack type and prompt type combination
    total_combinations = 0
    for attack_type, prompt_types in prompts_data.items():
        for prompt_type, prompt_data in prompt_types.items():
            total_combinations += 1

    print(f"Found {total_combinations} prompt combinations to process")

    for attack_type, prompt_types in prompts_data.items():
        for prompt_type, prompt_data in prompt_types.items():
            prompt_text = prompt_data.get("prompt", "")

            if not prompt_text:
                print(f"Warning: No prompt text found for {attack_type}/{prompt_type}")
                continue

            # Debug: Verify prompt text integrity
            debug_prompt_text(prompt_text)

            # Create output directory name within injected_pdfs folder
            main_output_dir = os.path.join("data", "injected_pdfs")
            sub_dir = f"{attack_type}_{prompt_type}_{injection_locus}"
            output_dir = os.path.join(main_output_dir, sub_dir)

            print(
                f"\nProcessing: {attack_type} -> {prompt_type} (injection: {injection_locus})"
            )
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            # Process each PDF file for this prompt combination
            successful_injections = 0
            failed_injections = 0

            print(f"Injecting prompt into {len(pdf_files)} PDF files...")

            for pdf_file in tqdm(pdf_files, desc=f"{attack_type}_{prompt_type}"):
                try:
                    # Get the filename without path
                    pdf_filename = os.path.basename(pdf_file)

                    # Create output path
                    output_path = os.path.join(output_dir, pdf_filename)

                    # Inject text into PDF
                    inject_text_into_pdf_silent(
                        pdf_file,
                        output_path,
                        prompt_text,
                        font_size,
                        font_path,
                        injection_locus,
                    )
                    successful_injections += 1

                except Exception as e:
                    tqdm.write(f"Failed to process {pdf_file}: {str(e)}")
                    failed_injections += 1

            print(
                f"Completed {attack_type}/{prompt_type}: {successful_injections} successful, {failed_injections} failed"
            )

    print("\nBatch injection complete!")
    print("All prompt combinations have been processed.")


def process_single_test(
    prompts_json_path, pdfs_dir, font_size=1.0, font_path=None, injection_locus="first"
):
    """
    Process a single random PDF with all prompts for testing purposes.
    """
    print(f"Reading prompts configuration from: {prompts_json_path}")
    prompts_data = read_prompts_json(prompts_json_path)
    if not prompts_data:
        return

    print(f"Scanning for PDF files in: {pdfs_dir}")
    pdf_pattern = os.path.join(pdfs_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    if not pdf_files:
        print(f"No PDF files found in '{pdfs_dir}'")
        return

    # Select one random PDF
    pdf_file = random.choice(pdf_files)
    print(f"Selected random PDF for testing: {pdf_file}")

    for attack_type, prompt_types in prompts_data.items():
        for prompt_type, prompt_data in prompt_types.items():
            prompt_text = prompt_data.get("prompt", "")
            if not prompt_text:
                print(f"Warning: No prompt text found for {attack_type}/{prompt_type}")
                continue

            main_output_dir = os.path.join("data", "injected_pdfs")
            sub_dir = f"test_{attack_type}_{prompt_type}_{injection_locus}"
            output_dir = os.path.join(main_output_dir, sub_dir)

            print(
                f"\nProcessing: {attack_type} -> {prompt_type} (injection: {injection_locus})"
            )
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

            pdf_filename = os.path.basename(pdf_file)
            output_path = os.path.join(output_dir, pdf_filename)

            try:
                inject_text_into_pdf_silent(
                    pdf_file,
                    output_path,
                    prompt_text,
                    font_size,
                    font_path,
                    injection_locus,
                )
                print(f"Successfully created test file: {output_path}")
            except Exception as e:
                print(f"Failed to process {pdf_file} for test: {str(e)}")

    print("\nSingle PDF test injection complete!")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Inject invisible text into PDFs.")
    parser.add_argument(
        "--prompts-json-path",
        type=str,
        default=os.path.join("data", "prompts", "prompts.json"),
        help="Path to the prompts JSON file.",
    )
    parser.add_argument(
        "--pdfs-dir",
        type=str,
        default=os.path.join("data", "redacted_pdfs"),
        help="Directory containing the PDF files.",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=1.0,
        help="Font size for the injected text.",
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=os.path.join("data", "fonts", "dejavusans.ttf"),
        help="Path to the font file.",
    )
    parser.add_argument(
        "--injection-locus",
        type=str,
        choices=["first", "last"],
        default="first",
        help="Injection locus: 'first' for first page top, 'last' for last page bottom.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "test"],
        default="batch",
        help="Run in 'batch' mode (all PDFs) or 'test' mode (one random PDF).",
    )
    args = parser.parse_args()

    print("Starting PDF injection process...")
    print(f"Prompts file: {args.prompts_json_path}")
    print(f"PDFs directory: {args.pdfs_dir}")
    print(f"Font size: {args.font_size}")
    print(f"Using font: {args.font_path}")
    print(f"Injection locus: {args.injection_locus}")
    print(f"Running in {args.mode} mode.")

    if args.mode == "batch":
        # Run batch processing
        process_batch_injection(
            args.prompts_json_path,
            args.pdfs_dir,
            args.font_size,
            args.font_path,
            args.injection_locus,
        )
    elif args.mode == "test":
        process_single_test(
            args.prompts_json_path,
            args.pdfs_dir,
            args.font_size,
            args.font_path,
            args.injection_locus,
        )

if __name__ == "__main__":
    main()
