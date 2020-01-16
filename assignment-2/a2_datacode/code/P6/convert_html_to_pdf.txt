import pdfkit

if __name__ == '__main__':
    pdfkit.from_file('image_score_html.html', 'test_image.pdf')
