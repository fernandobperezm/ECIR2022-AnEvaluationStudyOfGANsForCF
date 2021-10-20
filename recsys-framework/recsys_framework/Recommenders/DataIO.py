#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 27/04/2019

@author: Maurizio Ferrari Dacrema
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import zipfile
from enum import Enum
from typing import Any, Type

import numpy as np
import pandas as pd
import scipy.sparse as sps
from pandas import DataFrame


def attach_to_extended_json_decoder(enum_class: Type[Enum]):
    """
    Usage
    -----
    from recsys_framework.Recommenders.DataIO import attach_to_extended_json_decode
    import enum from Enum"

    @attach_to_extended_json_decoder
    class MyEnum(Enum):
        ...
    """
    ExtendedJSONDecoder.attach_enum(
        enum_class=enum_class
    )
    return enum_class


class ExtendedJSONDecoder(json.JSONEncoder):
    """
    Some values cannot be easily serialized/deserialized in JSON. For instance, enums and some
    numpy types.

    ExtendedJSONDecoder serves as a class to serialize/deserialize those values. The values exported
    by the class might not be portable across architectures, OS, or even programming languages.
    Nonetheless, it provides a very helpful way to serialize objects that otherwise might not be
    saved.
    """
    # _PUBLIC_ENUMS = {
    #     # **CFGAN_ENUMS,
    #     "EuclideanSimilarityFromDistanceMode": EuclideanSimilarityFromDistanceMode,
    #     "FeatureWeightingFunction": FeatureWeightingFunction,
    #     "SimilarityFunction": SimilarityFunction,
    # }

    _ATTACHED_ENUMS: dict[str, Type[Enum]] = dict()

    @staticmethod
    def attach_enum(enum_class: Type[Enum]):
        """
        Usage
        -----
        from recsys_framework.Recommenders.DataIO import ExtendedJSONDecoder
        import enum from Enum"

        class MyEnum(Enum):
            ...

        ExtendedJSONDecoder.attach_enum(enum_class=MyEnum)
        """

        enum_name = enum_class.__name__
        if enum_name in ExtendedJSONDecoder._ATTACHED_ENUMS:
            raise KeyError(
                f"Enum '{enum_name}' has already been attached. This may indicate that you attached this enum before "
                f"or another Enum has this name."
            )

        ExtendedJSONDecoder._ATTACHED_ENUMS[enum_name] = enum_class

    def default(self, obj: Any) -> Any:
        """
        This is the method that is called when a Python object is being serialized, i.e.,
        when json.dump(..., cls=ExtendedJSONDecoder) or json.dumps(..., cls=ExtendedJSONDecoder)
        are called.
        """
        if isinstance(obj, (np.integer, np.int32)):
            return int(obj)
        if isinstance(obj, (np.bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return {"__np_ndarray__": obj.tolist()}
        if isinstance(obj, Enum):
            return {"__enum__": str(obj)}

        # We leave the default JSONEncoder to raise an exception if it cannot serialize something.
        return json.JSONEncoder.default(self, obj)

    @staticmethod
    def decode_hook(obj: Any) -> Any:
        """
        This is the method that is called when a str object is being deserialized, i.e.,
        when json.load(..., object_hook=ExtendedJSONDecoder.decode_hook) or
        json.loads(..., object_hook=ExtendedJSONDecoder.decode_hook) are called.

        Notice that we have to load in the class all the enums classes, if we do not do this,
        Python will not know how to instantiate every specific enum class (basically we need to
        import them and have them in scope so it can recognize the enum class).
        """
        if "__enum__" in obj:
            name, member = obj["__enum__"].split(".")

            if name not in ExtendedJSONDecoder._ATTACHED_ENUMS:
                raise KeyError(
                    f"This enum '{name}' has not been attached to the '{ExtendedJSONDecoder.__name__}' class. "
                    f"Check the decorator '{attach_to_extended_json_decoder.__name__}' or the static method "
                    f"'{ExtendedJSONDecoder.__name__}.{ExtendedJSONDecoder.attach_enum.__name__}' documentation to "
                    f"learn how to attach enumerators to this class."
                )

            return getattr(ExtendedJSONDecoder._ATTACHED_ENUMS[name], member)
        if "__np_ndarray__" in obj:
            return np.array(obj["__np_ndarray__"])
        else:
            return obj


class DataIO(object):
    """ DataIO"""

    _DEFAULT_TEMP_FOLDER = ".temp_DataIO_"

    # _MAX_PATH_LENGTH_LINUX = 4096
    _MAX_PATH_LENGTH_WINDOWS = 255

    def __init__(self, folder_path):
        super(DataIO, self).__init__()

        self._is_windows = platform.system() == "Windows"

        self.folder_path = folder_path
        self._key_string_alert_done = False

        # if self._is_windows:
        #     self.folder_path = "\\\\?\\" + self.folder_path


    def _print(self, message):
        print("{}: {}".format("DataIO", message))


    def _get_temp_folder(self, file_name):
        """
        Creates a temporary folder to be used during the data saving
        :return:
        """

        # Ignore the .zip extension
        file_name = file_name[:-4]

        current_temp_folder = "{}{}_{}_{}/".format(self.folder_path, self._DEFAULT_TEMP_FOLDER, os.getpid(), file_name)

        if os.path.exists(current_temp_folder):
            self._print("Folder {} already exists, could be the result of a previous failed save attempt or multiple saver are active in parallel. " \
            "Folder will be removed.".format(current_temp_folder))

            shutil.rmtree(current_temp_folder, ignore_errors=True)

        os.makedirs(current_temp_folder)

        return current_temp_folder


    def _check_dict_key_type(self, dict_to_save):
        """
        Check whether the keys of the dictionary are string. If not, transforms them into strings
        :param dict_to_save:
        :return:
        """

        all_keys_are_str = all(isinstance(key, str) for key in dict_to_save.keys())

        if all_keys_are_str:
            return dict_to_save

        if not self._key_string_alert_done:
            self._print("Json dumps supports only 'str' as dictionary keys. Transforming keys to string, note that this will alter the mapper content.")
            self._key_string_alert_done = True

        dict_to_save_key_str = {str(key):val for (key,val) in dict_to_save.items()}

        assert all(dict_to_save_key_str[str(key)] == val for (key,val) in dict_to_save.items()), \
            "DataIO: Transforming dictionary keys into strings altered its content. Duplicate keys may have been produced."

        return dict_to_save_key_str


    def save_data(self, file_name, data_dict_to_save):

        # If directory does not exist, create with .temp_model_folder
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        if file_name[-4:] != ".zip":
            file_name += ".zip"


        current_temp_folder = self._get_temp_folder(file_name)

        attribute_to_file_name = {}
        attribute_to_json_file = {}
        for attrib_name, attrib_data in data_dict_to_save.items():
            current_file_path = current_temp_folder + attrib_name

            if isinstance(attrib_data, DataFrame):
                attrib_data.to_csv(current_file_path + ".csv", index=False)
                attribute_to_file_name[attrib_name] = attrib_name + ".csv"

            elif isinstance(attrib_data, sps.spmatrix):
                sps.save_npz(current_file_path, attrib_data)
                attribute_to_file_name[attrib_name] = attrib_name + ".npz"

            elif isinstance(attrib_data, np.ndarray):
                # allow_pickle is FALSE to prevent using pickle and ensure portability
                np.save(current_file_path, attrib_data, allow_pickle=False)
                attribute_to_file_name[attrib_name] = attrib_name + ".npy"

            else:
                # Try to parse it as json, if it fails and the data is a dictionary, use another zip file
                try:
                    _ = json.dumps(attrib_data, cls=ExtendedJSONDecoder)
                    attribute_to_json_file[attrib_name] = attrib_data
                    attribute_to_file_name[attrib_name] = attrib_name + ".json"

                except TypeError:

                    if isinstance(attrib_data, dict):
                        dataIO = DataIO(folder_path = current_temp_folder)
                        dataIO.save_data(file_name = attrib_name, data_dict_to_save=attrib_data)
                        attribute_to_file_name[attrib_name] = attrib_name + ".zip"

                    else:
                        raise TypeError(
                            "Type not recognized for attribute: {} with value {}".format(
                                attrib_name, attrib_data
                            )
                        )



        # Save list objects
        attribute_to_json_file[".DataIO_attribute_to_file_name"] = attribute_to_file_name.copy()

        for attrib_name, attrib_data in attribute_to_json_file.items():

            current_file_path = current_temp_folder + attrib_name
            attribute_to_file_name[attrib_name] = attrib_name + ".json"

            # if self._is_windows and len(current_file_path + ".json") >= self._MAX_PATH_LENGTH_WINDOWS:
            #     current_file_path = "\\\\?\\" + current_file_path

            absolute_path = current_file_path + ".json" if current_file_path.startswith(os.getcwd()) else os.getcwd() + current_file_path + ".json"

            assert not self._is_windows or (self._is_windows and len(absolute_path) <= self._MAX_PATH_LENGTH_WINDOWS), \
                "DataIO: Path of file exceeds {} characters, which is the maximum allowed under standard paths for Windows.".format(self._MAX_PATH_LENGTH_WINDOWS)


            with open(current_file_path + ".json", 'w') as outfile:

                if isinstance(attrib_data, dict):
                    attrib_data = self._check_dict_key_type(attrib_data)

                json.dump(attrib_data, outfile, cls=ExtendedJSONDecoder)



        with zipfile.ZipFile(self.folder_path + file_name, 'w', compression=zipfile.ZIP_DEFLATED) as myzip:

            for file_name in attribute_to_file_name.values():
                myzip.write(current_temp_folder + file_name, arcname = file_name)



        shutil.rmtree(current_temp_folder, ignore_errors=True)


    def load_data(self, file_name):

        if file_name[-4:] != ".zip":
            file_name += ".zip"

        dataFile = zipfile.ZipFile(self.folder_path + file_name)

        dataFile.testzip()

        current_temp_folder = self._get_temp_folder(file_name)

        try:

            try:
                attribute_to_file_name_path = dataFile.extract(".DataIO_attribute_to_file_name.json", path = current_temp_folder)
            except KeyError:
                attribute_to_file_name_path = dataFile.extract("__DataIO_attribute_to_file_name.json", path = current_temp_folder)


            with open(attribute_to_file_name_path, "r") as json_file:
                attribute_to_file_name = json.load(json_file,
                                                   object_hook=ExtendedJSONDecoder.decode_hook)

            data_dict_loaded = {}

            for attrib_name, file_name in attribute_to_file_name.items():

                attrib_file_path = dataFile.extract(file_name, path = current_temp_folder)
                attrib_data_type = file_name.split(".")[-1]

                if attrib_data_type == "csv":
                    attrib_data = pd.read_csv(attrib_file_path, index_col=False)

                elif attrib_data_type == "npz":
                    attrib_data = sps.load_npz(attrib_file_path)

                elif attrib_data_type == "npy":
                    # allow_pickle is FALSE to prevent using pickle and ensure portability
                    attrib_data = np.load(attrib_file_path, allow_pickle=False)

                elif attrib_data_type == "zip":
                    dataIO = DataIO(folder_path = current_temp_folder)
                    attrib_data = dataIO.load_data(file_name = file_name)

                elif attrib_data_type == "json":
                    with open(attrib_file_path, "r") as json_file:
                        attrib_data = json.load(json_file,
                                                object_hook=ExtendedJSONDecoder.decode_hook)

                else:
                    raise Exception("Attribute type not recognized for: '{}' of class: '{}'".format(attrib_file_path, attrib_data_type))

                data_dict_loaded[attrib_name] = attrib_data


        except Exception as exec:

            shutil.rmtree(current_temp_folder, ignore_errors=True)
            raise exec

        shutil.rmtree(current_temp_folder, ignore_errors=True)


        return data_dict_loaded
