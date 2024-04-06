from typing import Any, Iterator, Optional, TypeAlias

Word: TypeAlias = str | tuple[str, Any]
Words: TypeAlias = Iterator[Word]
Attributes: TypeAlias = Any
Record: TypeAlias = tuple[str, Attributes]
AdditionalAttributes: TypeAlias = Optional[dict[str, Any]]