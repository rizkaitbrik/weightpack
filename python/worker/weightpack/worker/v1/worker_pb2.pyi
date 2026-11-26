from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Codec(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CODEC_UNSPECIFIED: _ClassVar[Codec]
    CODEC_XOR: _ClassVar[Codec]
    CODEC_BIT1: _ClassVar[Codec]
    CODEC_SPARSE: _ClassVar[Codec]
    CODEC_LORA: _ClassVar[Codec]
CODEC_UNSPECIFIED: Codec
CODEC_XOR: Codec
CODEC_BIT1: Codec
CODEC_SPARSE: Codec
CODEC_LORA: Codec

class XORLayer(_message.Message):
    __slots__ = ("id", "blobs")
    ID_FIELD_NUMBER: _ClassVar[int]
    BLOBS_FIELD_NUMBER: _ClassVar[int]
    id: str
    blobs: _containers.RepeatedCompositeFieldContainer[Blob]
    def __init__(self, id: _Optional[str] = ..., blobs: _Optional[_Iterable[_Union[Blob, _Mapping]]] = ...) -> None: ...

class EncodeLayerRequest(_message.Message):
    __slots__ = ("codec", "base", "parents", "target_files", "out_dir")
    CODEC_FIELD_NUMBER: _ClassVar[int]
    BASE_FIELD_NUMBER: _ClassVar[int]
    PARENTS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FILES_FIELD_NUMBER: _ClassVar[int]
    OUT_DIR_FIELD_NUMBER: _ClassVar[int]
    codec: Codec
    base: _containers.RepeatedCompositeFieldContainer[Blob]
    parents: _containers.RepeatedCompositeFieldContainer[XORLayer]
    target_files: _containers.RepeatedScalarFieldContainer[str]
    out_dir: str
    def __init__(self, codec: _Optional[_Union[Codec, str]] = ..., base: _Optional[_Iterable[_Union[Blob, _Mapping]]] = ..., parents: _Optional[_Iterable[_Union[XORLayer, _Mapping]]] = ..., target_files: _Optional[_Iterable[str]] = ..., out_dir: _Optional[str] = ...) -> None: ...

class EncodedTensor(_message.Message):
    __slots__ = ("name", "file")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    name: str
    file: str
    def __init__(self, name: _Optional[str] = ..., file: _Optional[str] = ...) -> None: ...

class EncodeLayerResponse(_message.Message):
    __slots__ = ("tensors", "unchanged")
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    UNCHANGED_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.RepeatedCompositeFieldContainer[EncodedTensor]
    unchanged: int
    def __init__(self, tensors: _Optional[_Iterable[_Union[EncodedTensor, _Mapping]]] = ..., unchanged: _Optional[int] = ...) -> None: ...

class EvaluateRequest(_message.Message):
    __slots__ = ("image_ref", "suite")
    IMAGE_REF_FIELD_NUMBER: _ClassVar[int]
    SUITE_FIELD_NUMBER: _ClassVar[int]
    image_ref: str
    suite: str
    def __init__(self, image_ref: _Optional[str] = ..., suite: _Optional[str] = ...) -> None: ...

class EvaluateResponse(_message.Message):
    __slots__ = ("suite", "score")
    SUITE_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    suite: str
    score: float
    def __init__(self, suite: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class InfoRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class InfoResponse(_message.Message):
    __slots__ = ("version", "codecs")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    CODECS_FIELD_NUMBER: _ClassVar[int]
    version: str
    codecs: _containers.RepeatedScalarFieldContainer[Codec]
    def __init__(self, version: _Optional[str] = ..., codecs: _Optional[_Iterable[_Union[Codec, str]]] = ...) -> None: ...

class Blob(_message.Message):
    __slots__ = ("digest", "media_type", "path")
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    MEDIA_TYPE_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    digest: str
    media_type: str
    path: str
    def __init__(self, digest: _Optional[str] = ..., media_type: _Optional[str] = ..., path: _Optional[str] = ...) -> None: ...

class LoRALayer(_message.Message):
    __slots__ = ("id", "scaling", "blobs")
    ID_FIELD_NUMBER: _ClassVar[int]
    SCALING_FIELD_NUMBER: _ClassVar[int]
    BLOBS_FIELD_NUMBER: _ClassVar[int]
    id: str
    scaling: float
    blobs: _containers.RepeatedCompositeFieldContainer[Blob]
    def __init__(self, id: _Optional[str] = ..., scaling: _Optional[float] = ..., blobs: _Optional[_Iterable[_Union[Blob, _Mapping]]] = ...) -> None: ...

class SquashRequest(_message.Message):
    __slots__ = ("base", "layers", "out_dir", "xor_layers")
    BASE_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    OUT_DIR_FIELD_NUMBER: _ClassVar[int]
    XOR_LAYERS_FIELD_NUMBER: _ClassVar[int]
    base: _containers.RepeatedCompositeFieldContainer[Blob]
    layers: _containers.RepeatedCompositeFieldContainer[LoRALayer]
    out_dir: str
    xor_layers: _containers.RepeatedCompositeFieldContainer[XORLayer]
    def __init__(self, base: _Optional[_Iterable[_Union[Blob, _Mapping]]] = ..., layers: _Optional[_Iterable[_Union[LoRALayer, _Mapping]]] = ..., out_dir: _Optional[str] = ..., xor_layers: _Optional[_Iterable[_Union[XORLayer, _Mapping]]] = ...) -> None: ...

class MergedTensor(_message.Message):
    __slots__ = ("name", "file")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_FIELD_NUMBER: _ClassVar[int]
    name: str
    file: str
    def __init__(self, name: _Optional[str] = ..., file: _Optional[str] = ...) -> None: ...

class SquashResponse(_message.Message):
    __slots__ = ("tensors",)
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    tensors: _containers.RepeatedCompositeFieldContainer[MergedTensor]
    def __init__(self, tensors: _Optional[_Iterable[_Union[MergedTensor, _Mapping]]] = ...) -> None: ...
