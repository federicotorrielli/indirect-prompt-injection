import io
import argparse
import os
import sys
from pypdf import PdfReader, PdfWriter, PageObject

from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# --- CORRECTED FUNCTION ---
def create_invisible_text_overlay(text_to_add, page_width, page_height, font_size, font_name):
    """
    Creates an in-memory PDF page with a block of invisible, wrapping text.
    This version uses the correct low-level PDF command for invisibility.
    """
    packet = io.BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))

    # --- THE FIX ---
    # Save the current graphics state.
    can.saveState()
    # Use a low-level PDF command to set the text render mode to 3 (invisible).
    # This is the correct way to apply this state before drawing a high-level Paragraph.
    can._code.append('3 Tr')
    # --- END OF FIX ---

    # Define the bounding box for our text paragraph (1-inch margins)
    margin = 1 * inch
    frame_width = page_width - 2 * margin
    frame_height = page_height - 2 * margin
    
    # Create a custom style for our paragraph
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        name='Invisible',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=font_size,
        leading=font_size * 1.2,
        textColor=colors.black,
    )

    # Replace newlines with <br/> tags for the Paragraph object
    text_with_breaks = text_to_add.replace('\n', '<br/>')
    
    # Create the Paragraph object
    p = Paragraph(text_with_breaks, style)
    
    # Wrap and draw the paragraph from the top of the page.
    w, h = p.wrap(frame_width, frame_height)
    y_position = page_height - margin - h
    p.drawOn(can, margin, y_position)

    # Restore the canvas state to normal
    can.restoreState()
    can.save()
    packet.seek(0)
    
    return PdfReader(packet)
# --- END CORRECTED FUNCTION ---


def inject_text_into_pdf(input_path, output_path, invisible_text, font_size, font_path=None):
    """
    Injects invisible text into the FIRST PAGE ONLY, ensuring it is the first
    content in the page's data stream.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input PDF file not found at '{input_path}'")
        return

    font_name_to_use = "Helvetica"
    if font_path:
        if not os.path.exists(font_path):
            print(f"Error: Font file not found at '{font_path}'")
            return
        font_name_to_use = "CustomUnicodeFont"
        try:
            pdfmetrics.registerFont(TTFont(font_name_to_use, font_path))
            print(f"Successfully registered Unicode font: {font_path}")
        except Exception as e:
            print(f"Error: Failed to register font. Is it a valid .ttf file? Details: {e}")
            return

    print(f"Reading original PDF: {input_path}")
    existing_pdf = PdfReader(input_path)
    output_writer = PdfWriter()

    if not existing_pdf.pages:
        print("Error: The input PDF appears to be empty or corrupted.")
        return

    original_first_page = existing_pdf.pages[0]
    page_width = original_first_page.mediabox.width
    page_height = original_first_page.mediabox.height

    print(f"Creating invisible, wrapping text overlay with font '{font_name_to_use}' (size {font_size})...")
    overlay_pdf = create_invisible_text_overlay(invisible_text, page_width, page_height, font_size, font_name_to_use)
    overlay_page = overlay_pdf.pages[0]

    print("Injecting text as the first content element on page 1...")
    new_first_page = PageObject.create_blank_page(width=page_width, height=page_height)
    new_first_page.merge_page(overlay_page)
    new_first_page.merge_page(original_first_page)
    output_writer.add_page(new_first_page)

    if len(existing_pdf.pages) > 1:
        print(f"Copying the remaining {len(existing_pdf.pages) - 1} page(s)...")
        for i in range(1, len(existing_pdf.pages)):
            output_writer.add_page(existing_pdf.pages[i])

    print(f"Writing new PDF to: {output_path}")
    with open(output_path, "wb") as output_file:
        output_writer.write(output_file)
        
    print("\nInjection complete!")
    print(f"The new file '{output_path}' has been created with text injected only on the first page.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Injects a block of invisible, wrapping Unicode text onto the first page of a PDF.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", help="The invisible text you want to add.")
    group.add_argument("-tf", "--text-file", help="Path to a UTF-8 text file containing the message to inject.")

    parser.add_argument("-o", "--output", help="Path for the output PDF file.\nIf not provided, it will be named '<input_name>_injected.pdf'.")
    parser.add_argument("-s", "--font-size", type=float, default=1.0, help="Font size for invisible text. Default is 1.0.")
    parser.add_argument("-f", "--font-path", help="Path to a .ttf font file for Unicode character support (e.g., DejaVuSans.ttf).")

    args = parser.parse_args()

    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                text_to_inject = f.read()
            print(f"Successfully read text from file: {args.text_file}")
        except FileNotFoundError:
            print(f"Error: Text file not found at '{args.text_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading text file: {e}")
            sys.exit(1)
    else:
        text_to_inject = args.text
    
    if args.font_path is None and any(ord(c) > 127 for c in text_to_inject):
        print("\nWarning: Your text contains non-ASCII characters, but no --font-path was provided.")
        print("         The script may fail or produce incorrect output.")
        print("         Please provide a font like DejaVuSans.ttf using the -f flag.\n")

    if args.output:
        output_pdf_path = args.output
    else:
        base, ext = os.path.splitext(args.input_pdf)
        output_pdf_path = f"{base}_injected.pdf"

    inject_text_into_pdf(args.input_pdf, output_pdf_path, text_to_inject, args.font_size, args.font_path)
