from __future__ import annotations

import copy
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterator


class ConfigGrid:
    """
    Convert config.json into a grid a configuration combination.
    
    (1) GRID_KEY __grid__ must be the only key in the corresponding dict.
    
    (2) Following the GRID_KEY, a list [] telling things needed to be combined.
    
    (3) If an ensemble is to be combined, put it as a dict in the above list.
    
    Example:
    
    {
        
      "domain_1":{ 
        "__grid__":[
        {
            "domain_1-1": "shields",
            "domain_1-2": false
        },
        {
            "domain_1-1": "bh_1",
            "domain_1-2": false
        }
        ]
      },
    
      "domain_2": {"__grid__": [1,2,3]}
    
    }

    """

    GRID_KEY = "__grid__"

    def __init__(self, config: dict[str, Any]):
        self.raw_config = copy.deepcopy(config)

        self.fixed_params: dict[str, Any] = {}
        self.param_grid: dict[str, list[Any]] = {}

        self._parse(
            source=self.raw_config,
            fixed_target=self.fixed_params,
        )

    @classmethod
    def from_json(
        cls,
        config_path: str | Path,
    ) -> "ConfigGrid":
        config_path = Path(config_path).expanduser()

        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        if not isinstance(config, dict):
            raise TypeError(
                "The top-level configuration must be a dictionary."
            )

        return cls(config)

    @classmethod
    def _is_grid(cls, value: Any) -> bool:
        return (
            isinstance(value, dict)
            and set(value.keys()) == {cls.GRID_KEY}
        )

    def _parse(
        self,
        source: dict[str, Any],
        fixed_target: dict[str, Any],
        parent_path: str = "",
    ) -> None:
        """
        Seperate fixed and combinatirial params by recursion

        params seperated by dot:
            model.kernel
            bo.acquisition
        """

        for key, value in source.items():
            path = f"{parent_path}.{key}" if parent_path else key

            if self._is_grid(value):
                candidates = value[self.GRID_KEY]

                if not isinstance(candidates, list):
                    raise TypeError(
                        f"{path}.{self.GRID_KEY} must be a list."
                    )

                if not candidates:
                    raise ValueError(
                        f"{path}.{self.GRID_KEY} cannot be empty."
                    )

                self.param_grid[path] = copy.deepcopy(candidates)

            elif isinstance(value, dict):
                fixed_target[key] = {}

                self._parse(
                    source=value,
                    fixed_target=fixed_target[key],
                    parent_path=path,
                )

            else:
                fixed_target[key] = copy.deepcopy(value)

    @staticmethod
    def _set_nested(
        config: dict[str, Any],
        path: str,
        value: Any,
    ) -> None:

        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            current = current.setdefault(key, {})

        current[keys[-1]] = copy.deepcopy(value)

    def __len__(self) -> int:
        """total combination number"""
        

        if not self.param_grid:
            return 1

        return math.prod(
            len(values)
            for values in self.param_grid.values()
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """return a local context from global config grid"""

        if not isinstance(index, int):
            raise TypeError("ConfigGrid index must be an integer.")

        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(
                f"Configuration index {index} is out of range. "
                f"Valid range: 0-{len(self) - 1}."
            )

        context = copy.deepcopy(self.fixed_params)

        if not self.param_grid:
            return context

        parameter_names = list(self.param_grid)
        candidate_lists = [
            self.param_grid[name]
            for name in parameter_names
        ]

        selected_values = next(
            itertools.islice(
                itertools.product(*candidate_lists),
                index,
                index + 1,
            )
        )

        for path, value in zip(
            parameter_names,
            selected_values,
        ):
            self._set_nested(
                config=context,
                path=path,
                value=value,
            )

        return context

    def __iter__(self) -> Iterator[dict[str, Any]]:

        for index in range(len(self)):
            yield self[index]