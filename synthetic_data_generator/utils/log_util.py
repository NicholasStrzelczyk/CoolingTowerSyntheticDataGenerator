import logging


def setup_basic_logger(file_path):
    logging.basicConfig(
        filename=file_path,
        filemode='w',
        format="%(message)s",
        level=logging.DEBUG,
    )


def log(msg):
    logging.info(msg)


def log_and_print(msg):
    logging.info(msg)
    print(msg)
