from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)

    def add_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def close(self):
        self.writer.close()
