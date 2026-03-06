import keras

from pquant.core.keras.layers import (
    apply_final_compression,
    call_post_round_functions,
    get_ebops,
    get_layer_keep_ratio,
    post_epoch_functions,
    post_pretrain_functions,
    pre_epoch_functions,
    pre_finetune_functions,
    save_weights_functions,
)


class PQuantCallback(keras.callbacks.Callback):
    """
    Keras callback equivalent of train_model().

    Call model.fit(epochs=callback.total_epochs, callbacks=[callback], ...).
    Phase boundaries:
      [0, pretraining_epochs)                          → pretraining
      [pretraining_epochs, pretraining_epochs + rounds*epochs) → main rounds
      [pretraining_epochs + rounds*epochs, total_epochs)       → fine-tuning
    """

    def __init__(
        self,
        config,
        log_ebops=True,
        log_keep_ratio=True,
        apply_final_compression=True,
        pretraining_epochs=None,
        epochs=None,
        fine_tuning_epochs=None,
    ):
        super().__init__()
        tc = config.training_parameters
        self.config = config
        self.pretraining_epochs = pretraining_epochs if pretraining_epochs is not None else tc.pretraining_epochs
        self.rounds = tc.rounds
        self.epochs_per_round = epochs if epochs is not None else tc.epochs
        self.fine_tuning_epochs = fine_tuning_epochs if fine_tuning_epochs is not None else tc.fine_tuning_epochs
        self.rewind = tc.rewind
        self.save_weights_epoch = tc.save_weights_epoch
        self.log_ebops = log_ebops
        self.log_keep_ratio = log_keep_ratio
        self.apply_final_compression = apply_final_compression

        self._main_end = self.pretraining_epochs + self.rounds * self.epochs_per_round
        self._stage = "pretrain" if self.pretraining_epochs > 0 else "train"

    @property
    def total_epochs(self):
        return self._main_end + self.fine_tuning_epochs

    def on_train_begin(self, logs=None):
        # post_pretrain_functions is always called; if there are no pretraining
        # epochs the transition happens immediately.
        if self.pretraining_epochs == 0:
            post_pretrain_functions(self.model, self.config)
            # pre_finetune_functions is also always called; if there are no
            # main or fine-tuning epochs the transition also happens now.
            if self.epochs_per_round == 0 and self.fine_tuning_epochs == 0:
                pre_finetune_functions(self.model)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.pretraining_epochs:
            pre_epoch_functions(self.model, epoch, self.pretraining_epochs)
        elif epoch < self._main_end:
            rel = epoch - self.pretraining_epochs
            r, e = divmod(rel, self.epochs_per_round)
            if r == 0 and e == self.save_weights_epoch:
                save_weights_functions(self.model)
            pre_epoch_functions(self.model, e, self.epochs_per_round)
        else:
            e = epoch - self._main_end
            if e == 0:
                pre_finetune_functions(self.model)
            pre_epoch_functions(self.model, e, self.fine_tuning_epochs)

    @property
    def stage(self):
        """Current training stage: 'pretrain', 'train', or 'finetune'."""
        return self._stage

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.pretraining_epochs:
            self._stage = "pretrain"
            e = epoch
            post_epoch_functions(self.model, e, self.pretraining_epochs)
            if e == self.pretraining_epochs - 1:
                post_pretrain_functions(self.model, self.config)
                if self.epochs_per_round == 0 and self.fine_tuning_epochs == 0:
                    pre_finetune_functions(self.model)
        elif epoch < self._main_end:
            self._stage = "train"
            rel = epoch - self.pretraining_epochs
            r, e = divmod(rel, self.epochs_per_round)
            post_epoch_functions(self.model, e, self.epochs_per_round)
            if e == self.epochs_per_round - 1:
                call_post_round_functions(self.model, self.rewind, self.rounds, r)
                if r == self.rounds - 1 and self.fine_tuning_epochs == 0:
                    pre_finetune_functions(self.model)
        else:
            self._stage = "finetune"
            e = epoch - self._main_end
            post_epoch_functions(self.model, e, self.fine_tuning_epochs)
        if logs is not None:
            logs["stage"] = self._stage
        if self.log_ebops:
            logs["ebops"] = get_ebops(self.model)
        if self.log_keep_ratio:
            logs["remaining_weights"] = get_layer_keep_ratio(self.model)

    def on_train_end(self, logs=None):  # noqa: ARG002
        if self.apply_final_compression:
            apply_final_compression(self.model)


def train_model(model, config, train_func, valid_func, **kwargs):
    """
    Generic training loop, user provides training and validation functions
    """
    epoch = keras.ops.convert_to_tensor(0)  # Keeps track of all the epochs completed
    training_config = config.training_parameters
    if training_config.pretraining_epochs > 0:
        for e in range(training_config.pretraining_epochs):
            pre_epoch_functions(model, e, training_config.pretraining_epochs)
            train_func(model, epoch=epoch, **kwargs)
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config.pretraining_epochs)
            epoch += 1
    post_pretrain_functions(model, config)
    for r in range(training_config.rounds):
        for e in range(training_config.epochs):
            if r == 0 and training_config.save_weights_epoch == e:
                save_weights_functions(model)
            pre_epoch_functions(model, e, training_config.epochs)
            train_func(model, epoch=epoch, **kwargs)
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config.epochs)
            epoch += 1
        call_post_round_functions(model, training_config.rewind, training_config.rounds, r)
    pre_finetune_functions(model)
    if training_config.fine_tuning_epochs > 0:
        for e in range(training_config.fine_tuning_epochs):
            pre_epoch_functions(model, e, training_config.fine_tuning_epochs)
            train_func(model, epoch=epoch, **kwargs)
            valid_func(model, epoch=epoch, **kwargs)
            post_epoch_functions(model, e, training_config.fine_tuning_epochs)
            epoch += 1
    return model
