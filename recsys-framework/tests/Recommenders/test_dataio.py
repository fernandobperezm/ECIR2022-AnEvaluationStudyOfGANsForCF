from pathlib import Path

import pytest

from recsys_framework.Recommenders.DataIO import DataIO, ExtendedJSONDecoder, attach_to_extended_json_decoder
import numpy as np
import json

from enum import Enum


@pytest.fixture(autouse=True)
def initialize_extended_json_encoder():
    ExtendedJSONDecoder._ATTACHED_ENUMS = dict()


@pytest.fixture
def attach_enum_string_decorator() -> None:
    attach_to_extended_json_decoder(
        enum_class=EnumString
    )


@pytest.fixture
def attach_enum_int_decorator() -> None:
    attach_to_extended_json_decoder(
        enum_class=EnumInt
    )


@pytest.fixture
def attach_enum_string_static_method() -> None:
    ExtendedJSONDecoder.attach_enum(
        enum_class=EnumString
    )


@pytest.fixture
def attach_enum_int_static_method() -> None:
    ExtendedJSONDecoder.attach_enum(
        enum_class=EnumInt
    )


class EnumString(Enum):
    ONE = "ONE"
    TWO = "TWO"
    THREE = "THREE"


class EnumInt(Enum):
    ONE = 1
    TWO = 2
    THREE = 3


class TestDataIO:
    TEST_FILENAME = "tmp_save_data_metadata.zip"
    TEST_DICT_TO_SAVE = {"int": 5,
                         "str": "str",
                         "list": [1, 2, 3, 4, 5],
                         "np.int": 42,
                         "np.int32": np.int32(42),
                         "np.int64": np.int64(42),
                         "np.bool": True,
                         "np.bool_": np.bool_(False),
                         "np.array": np.array([[1, 2, 3], [4, 5, 6]]),
                         }

    def test_save_data(self, tmp_path: Path) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        # Everything is arranged in the class definition

        # Act
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Assert
        pass

    def test_load_data_no_enums(self, tmp_path) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        test_filename = "tmp_save_data_metadata.zip"
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Act
        data = data_io.load_data(file_name=test_filename)

        # Assert
        assert len(data.keys()) == len(self.TEST_DICT_TO_SAVE.keys())

        for k, v in self.TEST_DICT_TO_SAVE.items():
            assert k in data
            if isinstance(v, (list, np.ndarray)):
                assert np.all(np.equal(v, data[k]))
            else:
                assert v == data[k]

    def test_load_data_enum_str_attached_with_decorator(self, tmp_path: Path, attach_enum_string_decorator) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        test_filename = "tmp_save_data_metadata.zip"
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Act
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data = data_io.load_data(file_name=test_filename)

        # Assert
        assert len(data.keys()) == len(self.TEST_DICT_TO_SAVE.keys())

        for k, v in self.TEST_DICT_TO_SAVE.items():
            assert k in data
            if isinstance(v, (list, np.ndarray)):
                assert np.all(np.equal(v, data[k]))
            else:
                assert v == data[k]

    def test_load_data_enum_str_attached_with_static_method(self, tmp_path: Path, attach_enum_string_static_method) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        test_filename = "tmp_save_data_metadata.zip"
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Act
        data = data_io.load_data(file_name=test_filename)

        # Assert
        assert len(data.keys()) == len(self.TEST_DICT_TO_SAVE.keys())

        for k, v in self.TEST_DICT_TO_SAVE.items():
            assert k in data
            if isinstance(v, (list, np.ndarray)):
                assert np.all(np.equal(v, data[k]))
            else:
                assert v == data[k]

    def test_load_data_enum_int_attached_with_decorator(self, tmp_path: Path, attach_enum_int_decorator) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        test_filename = "tmp_save_data_metadata.zip"
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Act
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data = data_io.load_data(file_name=test_filename)

        # Assert
        assert len(data.keys()) == len(self.TEST_DICT_TO_SAVE.keys())

        for k, v in self.TEST_DICT_TO_SAVE.items():
            assert k in data
            if isinstance(v, (list, np.ndarray)):
                assert np.all(np.equal(v, data[k]))
            else:
                assert v == data[k]

    def test_load_data_enum_int_attached_with_static_method(self, tmp_path: Path, attach_enum_int_static_method) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        test_filename = "tmp_save_data_metadata.zip"
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Act
        data = data_io.load_data(file_name=test_filename)

        # Assert
        assert len(data.keys()) == len(self.TEST_DICT_TO_SAVE.keys())

        for k, v in self.TEST_DICT_TO_SAVE.items():
            assert k in data
            if isinstance(v, (list, np.ndarray)):
                assert np.all(np.equal(v, data[k]))
            else:
                assert v == data[k]

    def test_load_data_enum_str_not_attached(self, tmp_path: Path) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        self.TEST_DICT_TO_SAVE["str_enum"] = EnumString.ONE

        test_filename = "tmp_save_data_metadata.zip"
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Act
        with pytest.raises(KeyError):
            data = data_io.load_data(file_name=test_filename)

    def test_load_data_enum_int_not_attached(self, tmp_path: Path) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        self.TEST_DICT_TO_SAVE["int_enum"] = EnumInt.ONE

        test_filename = "tmp_save_data_metadata.zip"
        data_io = DataIO(folder_path=str(tmp_path.resolve()))
        data_io.save_data(file_name=self.TEST_FILENAME, data_dict_to_save=self.TEST_DICT_TO_SAVE)

        # Act
        with pytest.raises(KeyError):
            data = data_io.load_data(file_name=test_filename)

    def test_save_data_json_lib(self, ) -> None:
        """
        tmp_path is a temp Path object provided by pytest to run this test. Do not change the
        variable's name.
        """
        # Arrange
        test_filename = "tmp_save_data_metadata.zip"
        test_dict_to_save = {"int": 5,
                             "str": "str",
                             "list": [1, 2, 3, 4, 5],
                             "np.int": 42,
                             "np.int32": np.int32(42),
                             "np.int64": np.int64(42),
                             "np.bool": True,
                             "np.bool_": np.bool_(False),
                             #"np.array": np.array([[1, 2, 3], [4, 5, 6]]),
                             }

        # Act
        json_str = json.dumps(test_dict_to_save, cls=ExtendedJSONDecoder)

        # Assert
        assert len(json_str) > 0
