COMMAND TO RUN CODE: run code in this following order
1.cifar_finetune.py -> 2.create_image.py -> 3.create_html_table.py

Library to install:
1. matplotlib for showing/creating images
2. tabulate for create HTML table
3. pdfkit for converting HTML to pdf
4. natsort for sorting file names

Codes:
1. cifar_finetune.py for training/evaluating models
    -Flag to know
    1.1)USE_NEW_ARCHITECTURE = True, will add additional layer to the original files.
                                    it includes dropout rate and can specify number of
                                    hidden nodes.
    1.2)DO_TRAINING = True, to train additional layers added to the resnet20
    1.3)DO_TESTING = True, to test our model from training; SET TEST_BATCH_SIZE = 1 and SAVING_JSON = True
                          to test model on input one by one and create dictionaries of loss and score
                          to save in json format for later using in create images in HTML table.
    1.4)SAVING_JSON = True, to save dictionaries of loss and score from testing
                       !!!note: TEST_BATCH_SIZE in testing must be 1.
    -To get L2, set weight_decay > 0,
    -To get dropout, use new architecture and set dropout_rate > 0
2. create_image.py for creating individual test image saved into the specified folder.
3. create_html_table.py for creating HTML table for showing test images.
4. convert_html_to_pdf.py for converting html file to pdf file
