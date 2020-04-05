import fpdf


class PDFResults:

    def __init__(self):
        self.pdf = fpdf.FPDF(format='letter')  # pdf format
        self.pdf.add_font('DejaVu', '', 'output/DejaVuSansCondensed.ttf', uni=True)

    def append(self, text, align="L"):
        self.pdf.multi_cell(200, 10, txt=text, align=align)

    def append_heading(self, text):
        self.pdf.add_page()  # create new page
        self.pdf.set_font("DejaVu", size=14)  # font and textsize
        self.pdf.set_fill_color(200, 220, 255)
        self.pdf.cell(200, 10, txt=text, ln=1, align="C", fill=True)
        self.pdf.set_font("DejaVu", size=12)

    def append_results(self, text):
        self.pdf.set_text_color(0, 0, 255)
        self.pdf.multi_cell(200, 10, txt=text, align="L")
        self.pdf.set_text_color(0, 0, 0)

    def add_empty_line(self):
        self.pdf.ln(4)

    def print(self):
        self.pdf.output("output/results.pdf")
