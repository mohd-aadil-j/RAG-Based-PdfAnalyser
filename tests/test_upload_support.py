import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app


class UploadSupportTests(unittest.TestCase):
    def test_supported_file_types(self):
        self.assertTrue(app.is_supported_file_type("notes.pdf"))
        self.assertTrue(app.is_supported_file_type("notes.docx"))
        self.assertTrue(app.is_supported_file_type("slides.pptx"))
        self.assertTrue(app.is_supported_file_type("image.png"))
        self.assertTrue(app.is_supported_file_type("image.jpeg"))
        self.assertFalse(app.is_supported_file_type("sheet.xlsx"))


if __name__ == "__main__":
    unittest.main()
