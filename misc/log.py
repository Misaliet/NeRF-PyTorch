from tqdm import tqdm

def print_current_losses(log_name, i, loss, psnr):
    """print current losses on console; also save the losses to the disk
    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}"
    tqdm.write(message)

    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message