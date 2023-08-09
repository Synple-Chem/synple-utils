from typing import Callable, List, Tuple

from rdkit.Chem import MolFromSmiles, MolToSmiles
from vlib_enum.utils.enum_logger import get_logger

LOGGER = get_logger()


def ensure_readabilities_and_return_cano_smiles(
    mol_rprs: List[str],
    read_f: Callable = MolFromSmiles,
    bad_mol_includes: List[str] = [".", "[C]"],
) -> Tuple[List[int], List[str]]:
    """ensure readabilities and quality of mol representations and return canonical smiles

    Args:
        mol_rprs (List[str]): list of molrprs of molecules, e.g., smiles
        read_f (Callable, optional): mol read function. Defaults to MolFromSmiles.

    Returns:
        List[int]: index of valid molrprs
        List[str]: canonical smiles of valid molrprs
    """
    cano_idx: List[int] = []
    cano_smiles: List[str] = []
    for ii, molrpr in enumerate(mol_rprs):
        try:
            if is_bad_molrpr(molrpr, bad_mol_includes):
                raise ValueError(f"molrpr has {bad_mol_includes}")
            mol = read_f(molrpr)
            cano_smiles.append(MolToSmiles(mol))
            cano_idx.append(ii)
        except:
            LOGGER.warning(f"molrpr {molrpr} is not valid")
            continue
    return cano_idx, cano_smiles


def is_bad_molrpr(molrpr: str, include: List[str] = [".", "[C]"]) -> bool:
    """Check if the molrpr is includes invalid substrings

    Args:
        molrpr (str): molrpr to check, e.g., smiles
        include (List[str], optional): invalid substrings. Defaults to [".", "[C]"].

    Returns:
        bool: True if molrpr includes invalid substrings
    """
    for inc in include:
        if inc in molrpr:
            return True
    return False
