import abc
from typing import Dict, Type
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.base import BaseEstimator
from umap import UMAP


class DimPicker(abc.ABC):
    def __init__(self) -> None:
        """Base featurizer class"""
        super().__init__()

    def get_axis(self, raw: np.ndarray) -> np.ndarray:
        """Base method to compute fingerprints

        Args:
            raw (np.ndarray): raw features of the input data
        """
        raise NotImplementedError()

    def get_model(self) -> BaseEstimator:
        """Base method to return the DimPicker model"""
        raise NotImplementedError()

    def dim(self) -> int:
        """Size of the returned feature"""
        raise NotImplementedError()


AVAILABLE_DIM_PICKERS: Dict[str, Type[DimPicker]] = {}


def get_dim_picker(dimension_picker_name: str, **kwargs) -> DimPicker:
    """Basic factory function for dimension pickers"""
    return AVAILABLE_DIM_PICKERS[dimension_picker_name](**kwargs)


def register_dim_picker(name: str):
    """Decorator to register dimension picker classes"""

    def register_function(cls: Type[DimPicker]):
        if issubclass(cls, DimPicker):
            AVAILABLE_DIM_PICKERS[name] = cls
        else:
            raise ValueError("Not recognized descriptor type.")
        return cls

    return register_function


@register_dim_picker("pca")
class PCAPicker(DimPicker):
    def __init__(self, n_components: int = 3) -> None:
        """PCA dimension picker

        Args:
            n_components (int, optional): number of components. Defaults to 3.
        """
        super().__init__()
        self.n_components = n_components
        self.model = PCA(n_components=self.n_components)

    def get_axis(self, raw: np.ndarray) -> np.ndarray:
        """Get PCA axis

        Args:
            raw (np.ndarray): input data

        Returns:
            np.ndarray: PCA axis
        """
        self.model.fit(raw)
        return self.model.transform(raw)

    def get_model(self) -> BaseEstimator:
        """Get PCA model

        Returns:
            BaseEstimator: PCA model
        """
        return self.model

    def dim(self) -> int:
        """Size of the returned feature"""
        return self.n_components


@register_dim_picker("ica")
class ICAPicker(DimPicker):
    def __init__(self, n_components: int = 3) -> None:
        """ICA dimension picker

        Args:
            n_components (int, optional): number of components. Defaults to 3.
        """
        super().__init__()
        self.n_components = n_components
        self.model = FastICA(n_components=self.n_components)

    def get_axis(self, raw: np.ndarray) -> np.ndarray:
        """Get ICA axis

        Args:
            raw (np.ndarray): input data

        Returns:
            np.ndarray: ICA axis
        """
        self.model.fit(raw)
        return self.model.transform(raw)

    def get_model(self) -> BaseEstimator:
        """Get ICA model

        Returns:
            BaseEstimator: ICA model
        """
        return self.model

    def dim(self) -> int:
        """Size of the returned feature"""
        return self.n_components


@register_dim_picker("umap")
class UmapPicker(DimPicker):
    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 3,
        min_dist: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        """UMAP dimension picker

        Args:
            n_components (int, optional): number of components. Defaults to 3.
            n_neighbors (int, optional): number of neighbors. Defaults to 15.
            min_dist (float, optional): minimum distance. Defaults to 0.1.
            random_state (int | None, optional): random state. Defaults to None.
        """
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.random_state = random_state
        self.model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )

    def get_axis(self, raw: np.ndarray) -> np.ndarray:
        """Get UMAP axis

        Args:
            raw (np.ndarray): input data

        Returns:
            np.ndarray: UMAP axis
        """
        self.model.fit(raw)
        return self.model.transform(raw)

    def get_model(self) -> BaseEstimator:
        """Get UMAP model

        Returns:
            BaseEstimator: UMAP model
        """
        return self.model

    def dim(self) -> int:
        """Size of the returned feature"""
        return self.n_components
