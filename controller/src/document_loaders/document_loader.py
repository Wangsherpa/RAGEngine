import os
import fitz
from fastapi import UploadFile, HTTPException


class DocumentLoader:
    def __init__(self, pdf_processor: str = "pymupdf"):
        """Initializes the DocumentLoader with the specified PDF processor.

        Args:
            pdf_processor (str, optional): Tool to use for PDF text extraction ('pymupdf' or 'pdfminer'). Defaults to 'pymupdf'.
        """
        self.pdf_processor = pdf_processor

    def load_document(self, file: str):
        """Load the document based on its extension (either .txt or .pdf) and extract the text content.

        Args:
            file (str): The uploaded file.

        Raises:
            HTTPException: If an unsupported file type is uploaded.

        Returns:
            str: The content of the document.
        """
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension == ".txt":
            return self._load_txt(file)
        elif file_extension == ".pdf":
            return self._load_pdf(file)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload a .txt or .pdf file.",
            )

    async def _load_txt(self, file: UploadFile) -> str:
        """Process a .txt file and return its content as string.

        Args:
            file (UploadFile): The uploaded .txt file.

        Returns:
            str: Content of a text file.
        """
        content = (await file.read()).decode("utf-8")
        return content

    async def _load_pdf(self, file: UploadFile) -> str:
        """Process a .pdf file and return its content.

        Args:
            file (UploadFile): The uploaded .pdf file.

        Raises:
            HTTPException: If an unsupported processor is selected.

        Returns:
            str: Extracted text content from the PDF.
        """
        temp_filepath = f"temp_{file.filename}"
        with open(temp_filepath, "wb") as temp_file:
            temp_file.write(await file.read())

        if self.pdf_processor == "pymupdf":
            text = self._extract_text_pymupdf(temp_filepath)
        elif self.pdf_processor == "pdfminer":
            text = self._extract_text_pdfminer(temp_filepath)
        else:
            raise HTTPException(
                status_code=400, detail="Invalid PDF processor selected."
            )
        os.remove(temp_filepath)
        return text

    def _extract_text_pymupdf(self, file_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text content from the PDf file.
        """
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def _extract_text_pdfminer(self, file_path: str) -> str:
        """Extract text from a PDF file using pdfminer.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            str: Extracted text content from the PDf file.
        """
        # TODO: Needs to be implemented.
        pass
