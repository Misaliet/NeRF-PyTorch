from tqdm import tqdm

def print_current_losses(log_name, message):
    # tqdm.write(message)

    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message