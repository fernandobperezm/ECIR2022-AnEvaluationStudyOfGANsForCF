from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Type, Any

import attr
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from recsys_framework.Recommenders.DataIO import DataIO
from recsys_framework.Utils.conf_logging import get_logger

from conferences.cikm.cfgan.our_implementation.constants import CFGANActivation, CFGANOptimizer, CFGANMaskType, \
    CFGANMode
from conferences.cikm.cfgan.our_implementation.parameters import CFGANHyperParameters

tf.compat.v1.disable_v2_behavior()


logger = get_logger(__name__)


class CFGANModelAbstract(ABC):
    GENERATOR_MODEL_NAME = "GEN_MODEL"
    DISCRIMINATOR_MODEL_NAME = "DISC_MODEL"

    def __init__(
        self,
        urm_train: sp.csr_matrix,
        hyper_parameters: CFGANHyperParameters,
        num_training_item_weights_to_save: int = 0,
    ) -> None:
        super().__init__()

        self.hyper_parameters = hyper_parameters
        self.stats = {
            "generator_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.generator_steps)
            ),
            "generator_fake_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.generator_steps)
            ),
            "generator_zr_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.generator_steps)
            ),
            "generator_l2_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.generator_steps)
            ),
            "discriminator_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.discriminator_steps)
            ),
            "discriminator_real_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.discriminator_steps)
            ),
            "discriminator_fake_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.discriminator_steps)
            ),
            "discriminator_l2_losses": np.empty(
                shape=(self.hyper_parameters.epochs, self.hyper_parameters.discriminator_steps)
            ),
        }

        # Creates an array with epoch numbers. This array indicates in which epoch we will
        # save item weights.
        self._epochs_to_save_training_item_weights = np.linspace(
            start=0,
            stop=self.hyper_parameters.epochs,
            num=num_training_item_weights_to_save,
            endpoint=False,
            dtype=np.int32,
        )

        # self.training_item_weights: Dict[str, np.ndarray] = {}
        self.training_start_time: Optional[float] = None

        (
            self._dense_urm,
            self._num_rows_dataset,
            self._num_cols_dataset,
        ) = self._convert_urm_to_dataset(
            urm_train=urm_train
        )

        self._unseen_matrix = self._create_unseen_matrix()

        self._batch_list = np.arange(
            start=0,
            stop=self._num_rows_dataset,
        )

        (
            self._generator_num_input_features,
            self._generator_num_output_features,
            self._discriminator_num_input_features,
            self._discriminator_num_output_features
        ) = self._calculate_generator_discriminator_inputs_outputs()

        (
            self._generator_model_name,
            self._discriminator_model_name
        ) = self._get_generator_and_discriminator_names()

        self.sess: Optional[tf.compat.v1.Session] = None
        self.generator_saver: Optional[tf.compat.v1.train.Saver] = None
        self.discriminator_saver: Optional[tf.compat.v1.train.Saver] = None

    @staticmethod
    def get_optimizer(
        optimizer: CFGANOptimizer,
        learning_rate: float,
        loss: tf.compat.v1.Tensor,
        variables: List[tf.compat.v1.Variable],
    ) -> tf.compat.v1.Operation:
        if optimizer == CFGANOptimizer.ADAM:
            return tf.compat.v1.train.AdamOptimizer(
                learning_rate=learning_rate,
                epsilon=1e-8,  # In TFV1, this value defaults to 1e-8.
            ).minimize(
                loss=loss,
                var_list=variables,
            )
        else:
            raise ValueError(
                f"Optimizer {optimizer} not implemented. Valid values are: {list(CFGANOptimizer)}"
            )

    def cfgan_model(
        self,
        input_tensor: tf.compat.v1.Tensor,
        num_input_features: int,
        num_output_features: int,
        num_hidden_layers: int,
        num_hidden_features: int,
        activation: CFGANActivation,
        model_name: str,
        reuse: bool,
    ) -> Tuple[tf.compat.v1.Tensor, tf.compat.v1.Tensor]:
        # input->hidden
        layer_output, l2_norm, layer_weights, layer_bias = self.cfgan_dense_layer(
            input_tensor=input_tensor,
            num_input_features=num_input_features,
            num_output_features=num_hidden_features,
            activation=activation,
            model_name=model_name,
            layer_name=f"input",
            reuse=reuse,
        )

        # stacked hidden layers
        for hidden_layer_number in range(num_hidden_layers - 1):
            layer_output, l2_norm_hidden_layer, layer_weights, layer_bias = self.cfgan_dense_layer(
                input_tensor=layer_output,
                num_input_features=num_hidden_features,
                num_output_features=num_hidden_features,
                activation=activation,
                model_name=model_name,
                layer_name=f"hidden_{hidden_layer_number}",
                reuse=reuse,
            )
            l2_norm += l2_norm_hidden_layer

        # hidden -> output
        model_output, l2_norm_output, layer_weights, layer_bias = self.cfgan_dense_layer(
            input_tensor=layer_output,
            num_input_features=num_hidden_features,
            num_output_features=num_output_features,
            activation=None,
            model_name=model_name,
            layer_name=f"output",
            reuse=reuse,
        )
        l2_norm += l2_norm_output

        return (
            model_output,
            l2_norm,
        )

    @staticmethod
    def cfgan_dense_layer(
        input_tensor: tf.compat.v1.Tensor,
        num_input_features: int,
        num_output_features: int,
        activation: Optional[CFGANActivation],
        model_name: str,
        layer_name: str,
        reuse: bool,
    ) -> Tuple[tf.compat.v1.Tensor, tf.compat.v1.Tensor, tf.compat.v1.Variable, tf.compat.v1.Variable]:
        weights_name = f"{layer_name}/weights"
        bias_name = f"{layer_name}/bias"

        with tf.compat.v1.variable_scope(model_name) as scope:
            if reuse:
                scope.reuse_variables()

            layer_weights = tf.compat.v1.get_variable(
                weights_name,
                [num_input_features, num_output_features],
                initializer=tf.compat.v1.glorot_uniform_initializer(),
            )
            layer_bias = tf.compat.v1.get_variable(
                bias_name,
                [num_output_features],
                initializer=tf.compat.v1.random_uniform_initializer(-0.01, 0.01)
            )

            output_tensor = tf.compat.v1.matmul(input_tensor, layer_weights) + layer_bias

            l2_norm = tf.compat.v1.nn.l2_loss(layer_weights) + tf.compat.v1.nn.l2_loss(layer_bias)

            if activation is None:
                return tf.compat.v1.identity(output_tensor), l2_norm, layer_weights, layer_bias
            elif activation == CFGANActivation.SIGMOID:
                return tf.compat.v1.nn.sigmoid(output_tensor), l2_norm, layer_weights, layer_bias
            elif activation == CFGANActivation.TANH:
                return tf.compat.v1.nn.tanh(output_tensor), l2_norm, layer_weights, layer_bias
            else:
                raise ValueError(
                    f"Activation not accepted."
                )

    def calculate_discriminator_loss(
        self,
        discriminator_real_output: tf.compat.v1.Tensor,
        discriminator_fake_output: tf.compat.v1.Tensor,
        discriminator_variables: List[tf.compat.v1.Variable],
    ) -> Tuple[tf.compat.v1.Tensor, tf.compat.v1.Tensor, tf.compat.v1.Tensor, tf.compat.v1.Tensor]:
        """Calculates the loss for the discriminator.

        Notes
        -----
        It does not include the L2 loss because that one is calculated and automatically added to the training loss
         when the gradient is calculated/applied. If we do self.discriminator.losses here, it will duplicate the
         L2 loss in the gradient and will mess up the training.

        Args
        ----
        discriminator_real_output : tf.compat.v1.Tensor
         A (batch_size x 1) tensor which holds in each row the probability of D(x) being from
          the generator or not. real_output = D(x) where x = [c, urm], c is the condition
          vector, and  [,] is the append operation
        discriminator_fake_output : tf.compat.v1.Tensor
         A (batch_size x 1) tensor which holds in each row the probability of D(x) being from
          the generator or not. fake_output = D(x) where x = [c, G(c)], c is the condition
          vector, and  [,] is the append operation
        discriminator_variables : tf.compat.v1.Tensor
         Variables of the discriminator.
        """
        discriminator_real_loss: tf.compat.v1.Tensor = tf.compat.v1.reduce_mean(
            input_tensor=tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
                logits=discriminator_real_output,
                labels=tf.compat.v1.ones_like(discriminator_real_output)
            )
        )
        discriminator_fake_loss: tf.compat.v1.Tensor = tf.compat.v1.reduce_mean(
            input_tensor=tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
                logits=discriminator_fake_output,
                labels=tf.compat.v1.zeros_like(discriminator_fake_output)
            )
        )
        l2_loss: tf.compat.v1.Tensor = 0
        for discriminator_variable in discriminator_variables:
            l2_loss += tf.compat.v1.nn.l2_loss(discriminator_variable)
        l2_loss *= self.hyper_parameters.discriminator_regularization

        return (
            discriminator_real_loss + discriminator_fake_loss + l2_loss,
            discriminator_real_loss,
            discriminator_fake_loss,
            l2_loss,
        )

    def calculate_generator_loss(
        self,
        mask_zr: tf.compat.v1.Tensor,
        generator_output: tf.compat.v1.Tensor,
        generator_l2_loss: tf.compat.v1.Tensor,
        discriminator_fake_output: tf.compat.v1.Tensor,
        generator_variables: List[tf.compat.v1.Variable],
    ) -> Tuple[tf.compat.v1.Tensor, tf.compat.v1.Tensor, tf.compat.v1.Tensor, tf.compat.v1.Tensor]:
        """ Calculates the loss for the Generator.

        The loss for the generator is composed by several terms. The first, compares
        The loss is composed by several terms, i.e. loss = T1 + T2.

        T1 measures how much the generator could "fool" the discriminator, i.e., if the
        discriminator believes that the generator output comes from real data.

        T3 is an optional loss term that only applies if mask_type is ZERO_RECONSTRUCTION or
        ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING. In particular, T3 measures the MSE of the generator
        output with a random vector of non-interacted items. If mask_type is not of those mentioned
        above, then T3 is 0.

        Parameters
        ----------
        discriminator_fake_output
            A (batch_size, 1) tensor which holds in each row the probability of
            D(x) being from the generator or not. Mathematically speaking, fake_output = D(G(z)),
            where z is the combined random noise.
        generator_l2_loss
            asd
        generator_zr_loss
            asdasd

        Returns
        -------
        Tuple
            A tuple containing (T1 + T2, T1, T2)

        """
        generator_loss: tf.compat.v1.Tensor = tf.compat.v1.reduce_mean(
            tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(
                logits=discriminator_fake_output,
                labels=tf.compat.v1.ones_like(discriminator_fake_output),
            )
        )

        # Somehow this l2_loss_2 definition is not equal to l2_loss. Therefore, it cannot be used at the moment.
        # l2_loss_2: tf.compat.v1.Operation = 0.0
        # for generator_variable in generator_variables:
        #     print(f"* generator_variable={generator_variable}")
        #     l2_loss_2 += tf.compat.v1.nn.l2_loss(generator_variable)
        # l2_loss_2 *= self.hyper_parameters.generator_regularization

        l2_loss = generator_l2_loss * self.hyper_parameters.generator_regularization

        zr_loss = tf.compat.v1.convert_to_tensor(0.0)
        if self.hyper_parameters.mask_type in [
            CFGANMaskType.ZERO_RECONSTRUCTION,
            CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING
        ]:
            zr_loss = self.hyper_parameters.coefficient_zero_reconstruction * tf.compat.v1.reduce_mean(
                input_tensor=tf.compat.v1.reduce_sum(
                    input_tensor=tf.compat.v1.square(generator_output - tf.compat.v1.constant(0.0)) * mask_zr,
                    axis=1,
                    keep_dims=True
                )
            )

        return (
            generator_loss + l2_loss + zr_loss,
            generator_loss,
            l2_loss,
            zr_loss
        )

    @abstractmethod
    def _calculate_generator_discriminator_inputs_outputs(
        self,
    ) -> Tuple[int, int, int, int]:
        pass

    @abstractmethod
    def _get_generator_and_discriminator_names(
        self
    ) -> Tuple[str, str]:
        pass

    def _get_model_filenames(
        self,
        folder_path: str,
        file_name: str,
    ) -> Tuple[str, str, str, str]:
        model_folder_path = os.path.join(
            folder_path,
            f"{file_name}_models",
            "",
        )

        generator_file_path = os.path.join(
            model_folder_path,
            self.GENERATOR_MODEL_NAME,
        )

        discriminator_file_path = os.path.join(
            model_folder_path,
            self.DISCRIMINATOR_MODEL_NAME,
        )

        metadata_file_name = "metadata.zip"

        return model_folder_path, generator_file_path, discriminator_file_path, metadata_file_name

    def _save_model(
        self,
        folder_path: str,
        file_name: str,
        generator_saver: tf.compat.v1.train.Saver,
        discriminator_saver: tf.compat.v1.train.Saver,
        tf_session: tf.compat.v1.Session,
    ) -> None:
        (
            model_folder_path,
            generator_file_path,
            discriminator_file_path,
            metadata_file_name
        ) = self._get_model_filenames(
            folder_path=folder_path,
            file_name=file_name
        )

        generator_path_prefix = generator_saver.save(
            sess=tf_session,
            save_path=generator_file_path,
        )

        discriminator_path_prefix = discriminator_saver.save(
            sess=tf_session,
            save_path=discriminator_file_path,
        )

        data_io = DataIO(
            folder_path=model_folder_path
        )
        data_io.save_data(
            file_name=metadata_file_name,
            data_dict_to_save={
                "hyper_parameters": attr.asdict(self.hyper_parameters),
                "generator_path_prefix": generator_path_prefix,
                "discriminator_path_prefix": discriminator_path_prefix,
                **self.stats
            }
        )

        logger.info(f"Saved Generator to '{model_folder_path}' with the prefix '{self.GENERATOR_MODEL_NAME}'.")
        logger.info(f"Saved Discriminator to '{model_folder_path}' with the prefix '{self.DISCRIMINATOR_MODEL_NAME}'.")
        logger.info(f"Saved Metadata to '{model_folder_path} with the name {metadata_file_name}.")

    def _load_model(
        self,
        folder_path: str,
        file_name: str,
        hyper_parameter_cls: Type[Any],
        generator_saver: tf.compat.v1.train.Saver,
        discriminator_saver: tf.compat.v1.train.Saver,
        tf_session: tf.compat.v1.Session,
    ) -> None:
        (
            model_folder_path,
            generator_file_path,
            discriminator_file_path,
            metadata_file_name
        ) = self._get_model_filenames(
            folder_path=folder_path,
            file_name=file_name
        )

        data_io = DataIO(
            folder_path=model_folder_path
        )
        data_dict = data_io.load_data(
            file_name=metadata_file_name
        )

        generator_path_prefix = data_dict.pop(
            "generator_path_prefix"
        )
        discriminator_path_prefix = data_dict.pop(
            "discriminator_path_prefix"
        )

        self.hyper_parameters = hyper_parameter_cls(
            **data_dict.pop("hyper_parameters")
        )
        self.stats = {**data_dict}

        generator_saver.restore(
            sess=tf_session,
            save_path=generator_path_prefix,
        )

        discriminator_saver.restore(
            sess=tf_session,
            save_path=discriminator_path_prefix,
        )

        logger.info(f"Loaded Generator from '{generator_file_path}'.")
        logger.info(f"Loaded Discriminator from '{discriminator_file_path}'.")
        logger.info(f"Loaded Metadata from '{model_folder_path}{metadata_file_name}.")

    @abstractmethod
    def _get_generator_input(
        self,
        **kwargs,
    ) -> tf.compat.v1.Tensor:
        pass

    @abstractmethod
    def _get_discriminator_fake_and_real_input(
        self,
        condition: tf.compat.v1.Tensor,
        generator_output: tf.compat.v1.Tensor,
        mask_partial_masking: tf.compat.v1.Tensor,
        real_interactions: tf.compat.v1.Tensor,
    ) -> Tuple[tf.compat.v1.Tensor, tf.compat.v1.Tensor]:
        pass

    @abstractmethod
    def _train_step_discriminator(
        self,
        num_of_mini_batches: int,
        num_of_last_mini_batch: int,
        samples_partial_masking: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        pass

    @abstractmethod
    def _train_step_generator(
        self,
        num_of_mini_batches: int,
        num_of_last_mini_batch: int,
        samples_partial_masking: np.ndarray,
        samples_zero_reconstruction: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        pass

    @abstractmethod
    def get_item_weights(
        self,
        **kwargs,
    ) -> np.ndarray:
        pass

    def _convert_urm_to_dataset(
        self,
        urm_train: sp.csr_matrix,
    ) -> Tuple[np.ndarray, int, int]:
        if self.hyper_parameters.mode == CFGANMode.ITEM_BASED:
            dataset = urm_train.transpose().tocsr()
        elif self.hyper_parameters.mode == CFGANMode.USER_BASED:
            # In this mode, the URM_train must be kept as we received it.
            dataset = urm_train.tocsr()
        else:
            raise ValueError(
                f"Mode {self.hyper_parameters.mode} not implemented. Valid values are: {list(CFGANMode)}"
            )

        num_rows, num_cols = dataset.shape

        return (
            dataset.toarray(),
            num_rows,
            num_cols,
        )

    def _create_unseen_matrix(
        self,
    ) -> List[List[int]]:

        num_users, num_items = self._dense_urm.shape
        unseen_matrix = []

        for user_id in range(num_users):
            unseen_matrix.append(
                list(
                    np.where(self._dense_urm[user_id] == 0)[0]
                ),
            )

        return unseen_matrix

    def _calculate_epoch_zr_and_pm(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        zr_matrix = np.zeros_like(self._dense_urm)
        pm_matrix = self._dense_urm.copy()

        for row_idx, row_unseen_items in enumerate(self._unseen_matrix):
            num_unseen_items_in_row = len(row_unseen_items)

            if self.hyper_parameters.mask_type in [
                CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
                CFGANMaskType.ZERO_RECONSTRUCTION,
            ]:
                zr_samples = np.random.choice(
                    a=row_unseen_items,
                    size=int(num_unseen_items_in_row * self.hyper_parameters.ratio_zero_reconstruction // 100),
                    replace=False
                )
                zr_matrix[row_idx, zr_samples] = 1

            if self.hyper_parameters.mask_type in [
                CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING,
                CFGANMaskType.PARTIAL_MASKING,
            ]:
                pm_samples = np.random.choice(
                    a=row_unseen_items,
                    size=int(num_unseen_items_in_row * self.hyper_parameters.ratio_partial_masking // 100),
                    replace=False
                )
                pm_matrix[row_idx, pm_samples] = 1

        return (
            zr_matrix,
            pm_matrix,
        )

    def run_all_epochs(
        self
    ) -> None:
        for epoch in range(self.hyper_parameters.epochs):
            self.run_epoch(
                epoch=epoch,
            )

    def run_epoch(
        self,
        epoch: int
    ) -> None:
        if epoch == 0:
            self.training_start_time = time.time()

        if epoch in self._epochs_to_save_training_item_weights:
            self.stats[f"training_item_weights_epoch_{epoch}"] = self.get_item_weights(
                urm=None,
            )

        epoch_start_time = time.time()

        (
            samples_zero_reconstruction,
            samples_partial_masking
        ) = self._calculate_epoch_zr_and_pm()

        num_of_mini_batches = int(len(self._batch_list) / self.hyper_parameters.discriminator_batch_size) + 1
        num_of_last_mini_batch = len(self._batch_list) % self.hyper_parameters.discriminator_batch_size
        for discriminator_step in range(self.hyper_parameters.discriminator_steps):
            t1 = time.time()
            (
                discriminator_step_loss,
                discriminator_step_real_loss,
                discriminator_step_fake_loss,
                discriminator_step_sum_disc_loss,
            ) = self._train_step_discriminator(
                num_of_mini_batches=num_of_mini_batches,
                num_of_last_mini_batch=num_of_last_mini_batch,
                samples_partial_masking=samples_partial_masking,
            )

            self.stats["discriminator_losses"][epoch, discriminator_step] = discriminator_step_loss
            self.stats["discriminator_real_losses"][epoch, discriminator_step] = discriminator_step_real_loss
            self.stats["discriminator_fake_losses"][epoch, discriminator_step] = discriminator_step_fake_loss
            self.stats["discriminator_l2_losses"][epoch, discriminator_step] = discriminator_step_sum_disc_loss

            t2 = time.time()
            print(
                f"[{epoch + 1}: D][{discriminator_step + 1}] cost: {discriminator_step_loss:.4f}, within {t2 - t1} "
                f"seconds"
            )

        # G step
        num_of_mini_batches = int(len(self._batch_list) / self.hyper_parameters.generator_batch_size) + 1
        num_of_last_mini_batch = len(self._batch_list) % self.hyper_parameters.generator_batch_size
        for generator_step in range(self.hyper_parameters.generator_steps):
            t1 = time.time()
            (
                generator_step_loss,
                generator_step_fake_loss,
                generator_step_zr_loss,
                generator_step_l2_loss,

            ) = self._train_step_generator(
                num_of_mini_batches=num_of_mini_batches,
                num_of_last_mini_batch=num_of_last_mini_batch,
                samples_partial_masking=samples_partial_masking,
                samples_zero_reconstruction=samples_zero_reconstruction,
            )
            self.stats["generator_losses"][epoch, generator_step] = generator_step_loss
            self.stats["generator_fake_losses"][epoch, generator_step] = generator_step_fake_loss
            self.stats["generator_zr_losses"][epoch, generator_step] = generator_step_zr_loss
            self.stats["generator_l2_losses"][epoch, generator_step] = generator_step_l2_loss

            t2 = time.time()
            print(
                f"[{epoch + 1}: G][{generator_step + 1}] cost: {generator_step_loss:.4f}, within {t2 - t1} seconds"
            )

        if (self._epochs_to_save_training_item_weights.size > 0
            and epoch == self.hyper_parameters.epochs - 1):
            self.stats[f"training_item_weights_epoch_{self.hyper_parameters.epochs}"] = self.get_item_weights(
                urm=None
            )

        epoch_end_time = time.time()
        print(
            f"Finished epoch {epoch + 1} out of {self.hyper_parameters.epochs}. "
            f"Took {epoch_end_time - epoch_start_time:.2f}secs. "
        )
