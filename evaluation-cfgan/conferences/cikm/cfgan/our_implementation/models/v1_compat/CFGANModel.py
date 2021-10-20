from __future__ import annotations

import random
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from conferences.cikm.cfgan.our_implementation.constants import CFGANMaskType, CFGANMode
from conferences.cikm.cfgan.our_implementation.models.v1_compat.CFGANModelAbstract import CFGANModelAbstract
from conferences.cikm.cfgan.our_implementation.parameters import CFGANHyperParameters
from recsys_framework.Utils.conf_logging import get_logger


tf.compat.v1.disable_v2_behavior()


logger = get_logger(__name__)


class CFGANModel(CFGANModelAbstract):
    def __init__(
        self,
        urm_train: sp.csr_matrix,
        hyper_parameters: CFGANHyperParameters,
        initialize_model: bool,
        num_training_item_weights_to_save: int = 0,
    ) -> None:
        super().__init__(
            urm_train=urm_train,
            hyper_parameters=hyper_parameters,
            num_training_item_weights_to_save=num_training_item_weights_to_save,
        )
        with tf.compat.v1.Graph().as_default():
            if self.hyper_parameters.reproducibility_seed is not None:
                tf.compat.v1.set_random_seed(
                    seed=self.hyper_parameters.reproducibility_seed
                )
                np.random.seed(
                    seed=self.hyper_parameters.reproducibility_seed
                )
                random.seed(
                    self.hyper_parameters.reproducibility_seed
                )

            self.ph_condition = tf.compat.v1.placeholder(
                dtype=tf.compat.v1.float32,
                shape=(None, self._num_cols_dataset)
            )
            self.ph_real_interactions = tf.compat.v1.placeholder(
                dtype=tf.compat.v1.float32,
                shape=(None, self._num_cols_dataset)
            )
            self.ph_mask_partial_masking = tf.compat.v1.placeholder(
                dtype=tf.compat.v1.float32,
                shape=(None, self._num_cols_dataset)
            )
            self.ph_mask_zero_reconstruction = tf.compat.v1.placeholder(
                dtype=tf.compat.v1.float32,
                shape=(None, self._generator_num_output_features),
            )

            generator_input = self._get_generator_input(
                condition=self.ph_condition,
            )

            (
                self.generator_output,
                generator_l2_norm,
            ) = self.cfgan_model(
                input_tensor=generator_input,
                num_input_features=self._generator_num_input_features,
                num_output_features=self._generator_num_output_features,
                num_hidden_features=self.hyper_parameters.generator_hidden_features,
                num_hidden_layers=self.hyper_parameters.generator_hidden_layers,
                activation=self.hyper_parameters.generator_activation,
                model_name=self._generator_model_name,
                reuse=False,
            )

            (
                discriminator_fake_input,
                discriminator_real_input,
            ) = self._get_discriminator_fake_and_real_input(
                condition=self.ph_condition,
                generator_output=self.generator_output,
                mask_partial_masking=self.ph_mask_partial_masking,
                real_interactions=self.ph_real_interactions,
            )

            self.discriminator_real_output, _ = self.cfgan_model(
                input_tensor=discriminator_real_input,
                num_input_features=self._discriminator_num_input_features,
                num_output_features=self._discriminator_num_output_features,
                num_hidden_layers=self.hyper_parameters.discriminator_hidden_layers,
                num_hidden_features=self.hyper_parameters.discriminator_hidden_features,
                activation=self.hyper_parameters.discriminator_activation,
                model_name=self._discriminator_model_name,
                reuse=False,
            )
            self.discriminator_fake_output, _ = self.cfgan_model(
                input_tensor=discriminator_fake_input,
                num_input_features=self._discriminator_num_input_features,
                num_output_features=self._discriminator_num_output_features,
                num_hidden_layers=self.hyper_parameters.discriminator_hidden_layers,
                num_hidden_features=self.hyper_parameters.discriminator_hidden_features,
                activation=self.hyper_parameters.discriminator_activation,
                model_name=self._discriminator_model_name,
                reuse=True,
            )

            generator_variables = tf.compat.v1.get_collection(
                key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope=self._generator_model_name
            )
            discriminator_variables = tf.compat.v1.get_collection(
                key=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope=self._discriminator_model_name
            )

            (
                self.generator_loss,
                self.generator_fake_loss,
                self.generator_l2_loss,
                self.generator_zr_loss,
            ) = self.calculate_generator_loss(
                mask_zr=self.ph_mask_zero_reconstruction,
                generator_output=self.generator_output,
                discriminator_fake_output=self.discriminator_fake_output,
                generator_l2_loss=generator_l2_norm,
                generator_variables=generator_variables,
            )

            (
                self.discriminator_loss,
                self.discriminator_real_loss,
                self.discriminator_fake_loss,
                self.discriminator_l2_loss,
            ) = self.calculate_discriminator_loss(
                discriminator_fake_output=self.discriminator_fake_output,
                discriminator_real_output=self.discriminator_real_output,
                discriminator_variables=discriminator_variables,
            )

            self.generator_optimizer = self.get_optimizer(
                optimizer=self.hyper_parameters.generator_optimizer,
                learning_rate=self.hyper_parameters.generator_learning_rate,
                loss=self.generator_loss,
                variables=generator_variables,
            )

            self.discriminator_optimizer = self.get_optimizer(
                optimizer=self.hyper_parameters.discriminator_optimizer,
                learning_rate=self.hyper_parameters.discriminator_learning_rate,
                loss=self.discriminator_loss,
                variables=discriminator_variables,
            )

            self.generator_saver = tf.compat.v1.train.Saver(
                var_list=generator_variables,
            )
            self.discriminator_saver = tf.compat.v1.train.Saver(
                var_list=discriminator_variables,
            )

            config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})
            self.sess = tf.compat.v1.Session(config=config)

            if initialize_model:
                self.sess.run(tf.compat.v1.global_variables_initializer())

    def _calculate_generator_discriminator_inputs_outputs(
        self
    ) -> Tuple[int, int, int, int]:
        generator_num_input_features = self._num_cols_dataset
        generator_num_output_features = self._num_cols_dataset
        discriminator_num_input_features = self._num_cols_dataset * 2
        discriminator_num_output_features = 1

        return (
            generator_num_input_features,
            generator_num_output_features,
            discriminator_num_input_features,
            discriminator_num_output_features,
        )

    def _get_generator_and_discriminator_names(
        self
    ) -> Tuple[str, str]:

        generator_model_name = (
            f"TFV1_CFGAN_"
            f"{self.hyper_parameters.benchmark.value}_"
            f"{self.hyper_parameters.mode.value}_"
            f"{self.hyper_parameters.mask_type.value}_"
            f"generator"
        )

        discriminator_model_name = (
            f"TFV1_CFGAN_"
            f"{self.hyper_parameters.benchmark.value}_"
            f"{self.hyper_parameters.mode.value}_"
            f"{self.hyper_parameters.mask_type.value}_"
            f"discriminator"
        )

        return generator_model_name, discriminator_model_name

    def save_model(
        self,
        folder_path: str,
        file_name: str,
    ) -> None:
        super()._save_model(
            folder_path=folder_path,
            file_name=file_name,
            generator_saver=self.generator_saver,
            discriminator_saver=self.discriminator_saver,
            tf_session=self.sess,
        )

    def load_model(
        self,
        folder_path: str,
        file_name: str,
    ) -> None:
        super()._load_model(
            folder_path=folder_path,
            file_name=file_name,
            hyper_parameter_cls=CFGANHyperParameters,
            generator_saver=self.generator_saver,
            discriminator_saver=self.discriminator_saver,
            tf_session=self.sess,
        )

    def _get_generator_input(
        self,
        condition: tf.compat.v1.Tensor,
    ) -> tf.compat.v1.Tensor:
        return condition

    def _get_discriminator_fake_and_real_input(
        self,
        condition: tf.compat.v1.Tensor,
        generator_output: tf.compat.v1.Tensor,
        mask_partial_masking: tf.compat.v1.Tensor,
        real_interactions: tf.compat.v1.Tensor,
    ) -> Tuple[tf.compat.v1.Tensor, tf.compat.v1.Tensor]:

        if self.hyper_parameters.mask_type == CFGANMaskType.ZERO_RECONSTRUCTION:
            mask = real_interactions
        elif self.hyper_parameters.mask_type == CFGANMaskType.PARTIAL_MASKING:
            mask = mask_partial_masking
        elif self.hyper_parameters.mask_type == CFGANMaskType.ZERO_RECONSTRUCTION_AND_PARTIAL_MASKING:
            mask = mask_partial_masking
        else:
            raise ValueError("Invalid CFGANMask value.")

        fake_interactions = generator_output * mask

        discriminator_fake_input = tf.compat.v1.concat(
            values=[condition, fake_interactions],
            axis=1,
        )
        discriminator_real_input = tf.compat.v1.concat(
            values=[condition, real_interactions],
            axis=1,
        )

        return discriminator_fake_input, discriminator_real_input

    def get_item_weights(
        self,
        **kwargs,
    ) -> np.ndarray:
        item_weights: np.ndarray = self.sess.run(
            fetches=self.generator_output,
            feed_dict={
                self.ph_condition: self._dense_urm
            }
        )
        if self.hyper_parameters.mode == CFGANMode.ITEM_BASED:
            item_weights = np.transpose(item_weights)
        return item_weights

    def _train_step_discriminator(
        self,
        num_of_mini_batches: int,
        num_of_last_mini_batch: int,
        samples_partial_masking: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Training step for the CFGAN discriminator.

        It begins by calculating the condition to be used for both the generator and discriminator, then calculates the
        generator's input (which in this case is the condition itself). A mask is then applied to the generator's
        output. This mask can be either the real-profiles or the 'batch_mask_pm' tensor. Lastly, the discriminator is
        fed with this masked output and the real profile. From its outputs, the discriminator loss is calculated and
        the gradients applied.

        Args
        ----
        num_of_mini_batches : tf.compat.v1.keras.Sequential
         Interactions to be considered in this batch
        num_of_last_mini_batch : tf.compat.v1.keras.Sequential
         User indices to be considered in this batch.
        samples_partial_masking : tf.compat.v1.Tensor
         Random PM samples to be considered in this batch.
        """
        discriminator_step_loss = 0.0
        discriminator_step_real_loss = 0.0
        discriminator_step_fake_loss = 0.0
        discriminator_step_sum_disc_loss = 0.0

        random.shuffle(self._batch_list)

        for batch_id in range(num_of_mini_batches):
            start = batch_id * self.hyper_parameters.discriminator_batch_size

            if batch_id == num_of_mini_batches - 1:  # if it is the last minibatch
                num_of_batches = num_of_last_mini_batch
            else:
                num_of_batches = self.hyper_parameters.discriminator_batch_size
            end = start + num_of_batches

            batch_indices = self._batch_list[start:end]

            (
                _,
                discriminator_batch_loss,
                discriminator_batch_real_loss,
                discriminator_batch_fake_loss,
                discriminator_batch_l2_loss,
            ) = self.sess.run(
                fetches=[
                    self.discriminator_optimizer,
                    self.discriminator_loss,
                    self.discriminator_real_loss,
                    self.discriminator_fake_loss,
                    self.discriminator_l2_loss,
                ],
                feed_dict={
                    self.ph_real_interactions: self._dense_urm[batch_indices],
                    self.ph_mask_partial_masking: samples_partial_masking[batch_indices],
                    self.ph_condition: self._dense_urm[batch_indices],
                },
            )

            discriminator_step_loss += discriminator_batch_loss
            discriminator_step_real_loss += discriminator_batch_real_loss
            discriminator_step_fake_loss += discriminator_batch_fake_loss
            discriminator_step_sum_disc_loss += discriminator_batch_l2_loss

        return (
            discriminator_step_loss,
            discriminator_step_real_loss,
            discriminator_step_fake_loss,
            discriminator_step_sum_disc_loss,
        )

    def _train_step_generator(
        self,
        num_of_mini_batches: int,
        num_of_last_mini_batch: int,
        samples_partial_masking: np.ndarray,
        samples_zero_reconstruction: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Training step for the CFGAN generator.

        It begins by calculating the condition to be used for both the generator and discriminator, then calculates the
        generator's input (which in this case is the condition itself). Then, a mask is applied to the generator's
        output. This mask can be either the real-profiles or the 'batch_mask_pm' tensor. Lastly, the discriminator is
        fed with this masked output and then the generator loss is calculated, alongside the ZR loss if required.

        Args
        ----
        num_of_mini_batches
         Interactions to be considered in this batch
        num_of_last_mini_batch
         User indices to be considered in this batch.
        samples_zero_reconstruction
         Random ZR samples to be considered in this batch.
        samples_partial_masking
         Random PM samples to be considered in this batch.
        """
        generator_step_loss = 0.0
        generator_step_fake_loss = 0.0
        generator_step_zr_loss = 0.0
        generator_step_l2_loss = 0.0

        random.shuffle(self._batch_list)

        for batchId in range(num_of_mini_batches):
            start = batchId * self.hyper_parameters.generator_batch_size
            if batchId == num_of_mini_batches - 1:  # if it is the last minibatch
                num_of_batches = num_of_last_mini_batch
            else:
                num_of_batches = self.hyper_parameters.generator_batch_size
            end = start + num_of_batches
            batch_indices = self._batch_list[start:end]

            (
                _,
                generator_batch_loss,
                generator_batch_fake_loss,
                generator_batch_zr_loss,
                generator_batch_l2_loss,
            ) = self.sess.run(
                fetches=[
                    self.generator_optimizer,
                    self.generator_loss,
                    self.generator_fake_loss,
                    self.generator_zr_loss,
                    self.generator_l2_loss,
                ],
                feed_dict={
                    self.ph_condition: self._dense_urm[batch_indices],
                    self.ph_real_interactions: self._dense_urm[batch_indices],
                    self.ph_mask_partial_masking: samples_partial_masking[batch_indices],
                    self.ph_mask_zero_reconstruction: samples_zero_reconstruction[batch_indices],
                },
            )
            generator_step_loss += generator_batch_loss
            generator_step_fake_loss += generator_batch_fake_loss
            generator_step_zr_loss += generator_batch_zr_loss
            generator_step_l2_loss += generator_batch_l2_loss

        return (
            generator_step_loss,
            generator_step_fake_loss,
            generator_step_zr_loss,
            generator_step_l2_loss,
        )
