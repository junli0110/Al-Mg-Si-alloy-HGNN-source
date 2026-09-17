from __future__ import annotations

import argparse
import copy
import re
import json
import math
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ShuffleSplit, GroupKFold, GroupShuffleSplit
from tqdm.auto import tqdm


try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATv2Conv, HeteroConv, global_max_pool, global_mean_pool

    PYG_AVAILABLE = True
except ImportError:
    HeteroData = None
    DataLoader = None
    GATv2Conv = None
    HeteroConv = None
    global_max_pool = None
    global_mean_pool = None
    PYG_AVAILABLE = False


ALL_ELEMENTS = ["Al", "Mg", "Si", "Mn", "Cu", "Fe", "Cr", "Zr", "Ti", "Zn", "V", "Ni", "Sc", "Ag", "Er", "Y"]
SOLUTE_ELEMENTS = ALL_ELEMENTS[1:]

@dataclass
class TrainConfig:
    seed: int = 3407
    n_splits: int = 5
    validation_fraction: float = 0.15
    hidden_dim: int = 64
    heads: int = 4
    n_layers: int = 2
    dropout: float = 0.15
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 400
    patience: int = 45
    min_delta: float = 1e-4
    composition_round_decimals: int = 4
    num_workers: int = 0
    stage_aware_readout: bool = False
    descriptor_skip_readout: bool = False
    refit_full_outer_train: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def require_torch_geometric() -> None:
    if not PYG_AVAILABLE:
        raise ImportError(
            "torch-geometric is required for HeteroData conversion and HGNN training. "
            "Install the packages from requirements.txt."
        )


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"sample_no": str, "composition_id": str, "source_id": str})
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def infer_route(row: pd.Series) -> str:
    route = str(row.get("route_label", "")).strip().upper()
    if route not in {"T5", "T6"}:
        raise ValueError("Every route_label must be exactly T5 or T6.")
    return route


def numeric(value: object) -> float:
    result = pd.to_numeric(value, errors="coerce")
    return float(result) if pd.notna(result) else np.nan


def _normalized_text(value: object) -> str:
    return " ".join(str(value).lower().replace("α", "alpha").split())


def load_element_properties(
    path: Path,
    selected_property_names: Sequence[str],
    required_elements: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load the selected element descriptors."""

    table = read_table(path)
    if table.shape[1] < 2:
        raise ValueError("Element-property table must contain element names and properties.")

    first_col = table.columns[0]
    table = table.rename(columns={first_col: "element"})
    table["element"] = table["element"].astype(str).str.strip()
    table = table.set_index("element")

    if table.index.duplicated().any():
        raise ValueError("Duplicate element rows in property table.")
    for column in list(table.columns):
        match = re.match(r"^([ACEGS]\d+)(?:\s|$)", str(column))
        if match and match.group(1) not in table:
            values = pd.to_numeric(table[column], errors="raise")
            text = str(column).lower()
            if match.group(1) == "C21" and "mj/" in text:
                values = values / 1000.0
            if match.group(1) == "S15" and "(pm)" in text:
                values = values / 1000.0
            table[match.group(1)] = values
    selected_columns: List[str] = []
    selected_names: List[str] = []
    normalized_lookup = {_normalized_text(column): column for column in table.columns}
    for requested in selected_property_names:
        requested = str(requested).strip()
        exact = next((column for column in table.columns if str(column).strip() == requested), None)
        match = exact or normalized_lookup.get(_normalized_text(requested))
        if match is None:
            raise ValueError(f"Selected element property was not found in {path.name}: {requested}")
        selected_columns.append(match)
        selected_names.append(requested)

    if len(selected_columns) < 4:
        raise ValueError(
            "Fewer than four requested element descriptors were found. "
            "Inspect the element-property table before training."
        )

    required_elements = list(required_elements) if required_elements is not None else ALL_ELEMENTS
    missing_elements = [element for element in required_elements if element not in table.index]
    if missing_elements:
        raise ValueError(f"Missing element-property rows: {missing_elements}")
    selected = table.reindex(ALL_ELEMENTS)[selected_columns].apply(pd.to_numeric, errors="coerce")
    selected.columns = selected_names
    if selected.isna().all(axis=0).any():
        bad = selected.columns[selected.isna().all(axis=0)].tolist()
        raise ValueError(f"Selected element-property columns are entirely missing: {bad}")
    if not np.isfinite(selected.loc[required_elements].to_numpy(dtype=float)).all():
        raise ValueError("Required elemental descriptors contain missing/nonfinite values.")
    return selected, selected_names


def make_composition_group(row: pd.Series, decimals: int = 4) -> str:
    values = [numeric(row.get(element, 0.0)) for element in SOLUTE_ELEMENTS]
    values = [0.0 if np.isnan(value) else round(value, decimals) for value in values]
    return "|".join(f"{element}={value:.{decimals}f}" for element, value in zip(SOLUTE_ELEMENTS, values))


@dataclass
class MatrixStandardizer:
    median: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None

    def fit(self, matrices: Sequence[np.ndarray]) -> "MatrixStandardizer":
        values = np.vstack(matrices).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isnan(values), median, values)
        mean = filled.mean(axis=0)
        scale = filled.std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        self.median, self.mean, self.scale = median, mean, scale
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.median is None or self.mean is None or self.scale is None:
            raise RuntimeError("Standardizer is not fitted.")
        values = np.asarray(matrix, dtype=float)
        filled = np.where(np.isnan(values), self.median, values)
        return ((filled - self.mean) / self.scale).astype(np.float32)

    def state_dict(self) -> Dict[str, List[float]]:
        return {
            "median": self.median.tolist(),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }


@dataclass
class TargetStandardizer:
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "TargetStandardizer":
        self.mean = np.mean(values, axis=0)
        self.scale = np.std(values, axis=0)
        self.scale = np.where(self.scale < 1e-12, 1.0, self.scale)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def state_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


ELEMENT_FEATURES = {
    'UTS': ['C11', 'S15', 'S10', 'C14', 'C15', 'C21'],
    'EL': ['S15', 'S10', 'C14', 'C21', 'C13', 'S18'],
}
PHASE_DESCRIPTORS = {
    'f_alpha_EXT': ('EXT', 'Al15_FeMn3Si2', 'pd_EXT_f_Al15_FeMn3Si2'),
    'DF_alpha_SS': ('SS', 'Al15_FeMn3Si2', 'pd_SS_DF(@|Al15_FeMn3Si2)'),
    'G_FCC_EXT': ('EXT', 'Fcc', 'pd_EXT_G_Fcc'),
    'S_FCC_SS': ('SS', 'Fcc', 'pd_SS_S_Fcc'),
    'DF_beta_EXT': ('EXT', 'Mg2Si', 'pd_EXT_DF(@|Mg2Si)'),
    'H_Q_AA': ('AA', 'Q_Al5Cu2Mg8Si6', 'pd_AA_H_Q_Al5Cu2Mg8Si6'),
    'f_beta_AA': ('AA', 'Mg2Si', 'pd_AA_f_Mg2Si'),
    'f_Q_AA': ('AA', 'Q_Al5Cu2Mg8Si6', 'pd_AA_f_Q_Al5Cu2Mg8Si6'),
}
PHASE_FEATURES = {
    'UTS': list(PHASE_DESCRIPTORS),
    'EL': ['f_alpha_EXT', 'DF_alpha_SS', 'S_FCC_SS', 'H_Q_AA', 'f_beta_AA', 'f_Q_AA'],
}
PROCESS_ALIASES = {
    'EXS(m/min)': 'vEXT', 'EXR': 'REXT',
    'SS': 'TSS', 'SS-t': 'tSS', 'AA': 'TAA', 'AA-t': 'tAA',
}
BOUNDS = {'Mg': (.5, 1.4), 'Si': (.5, 1.4), 'Mn': (0, .7), 'Cu': (0, 1.1),
          'Zn': (0, .6), 'Cr': (0, .3), 'Zr': (0, .2), 'Fe': (0, .4),
          'TEXT': (480, 560), 'vEXT': (1, 7), 'REXT': (20, 60),
          'TSS': (520, 560), 'tSS': (5, 120), 'TAA': (160, 220), 'tAA': (2, 24)}

def alias_column(df, canonical, alias):
    if canonical in df and alias in df:
        a = pd.to_numeric(df[canonical], errors='coerce')
        b = pd.to_numeric(df[alias], errors='coerce')
        if not np.allclose(a, b, equal_nan=True):
            raise ValueError(f'Conflicting columns: {canonical} and {alias}')
    elif alias in df:
        df[canonical] = df[alias]

def normalize_inputs(df, solutes, phase_descriptors, target=None):
    """Normalize and validate composition, processing and phase inputs."""
    df = df.copy().reset_index(drop=True)
    if df.empty or df.columns.duplicated().any():
        raise ValueError('Input is empty or has duplicate columns.')
    if 'route_label' not in df:
        if 'route' in df:
            df['route_label'] = df['route']
        else:
            raise ValueError('Explicit route_label (T5/T6) is required.')
    df['route_label'] = df['route_label'].astype(str).str.strip().str.upper()
    if not df.route_label.isin(['T5', 'T6']).all():
        raise ValueError('Every route_label must be exactly T5 or T6.')
    if 'route' in df and not df['route'].astype(str).str.strip().str.upper().eq(df.route_label).all():
        raise ValueError('Conflicting route and route_label values.')
    if 'sample_no' not in df:
        raise ValueError('Provide a unique sample_no for each composition-process record.')
    if df.sample_no.isna().any() or df.sample_no.astype(str).str.strip().eq('').any() or df.sample_no.astype(str).str.strip().duplicated().any():
        raise ValueError('sample_no must be nonempty and unique.')
    df['sample_no'] = df.sample_no.astype(str).str.strip()
    unknown = [c for c in df if re.fullmatch(r'[A-Z][a-z]?', str(c)) and c not in ['Al', *solutes]]
    if unknown:
        raise ValueError(f'Unsupported elemental columns: {unknown}; extend the element registry explicitly.')
    for c in ['Mg', 'Si']:
        if c not in df: raise ValueError(f'Missing composition column {c}')
    for c in solutes:
        if c not in df: df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='raise')
        if not np.isfinite(df[c]).all() or (df[c] < 0).any():
            raise ValueError(f'{c} must contain finite, nonnegative wt.% values.')
    remainder = 100 - df[list(solutes)].sum(axis=1)
    if (remainder <= 0).any(): raise ValueError('Composition must leave a positive Al balance.')
    if 'Al' in df and not np.allclose(pd.to_numeric(df.Al, errors='raise'), remainder, atol=1e-4):
        raise ValueError('Supplied Al does not agree with the wt.% balance.')
    df['Al'] = remainder
    for old, new in PROCESS_ALIASES.items(): alias_column(df, old, new)
    required = ['TEXT', 'EXS(m/min)', 'EXR', 'AA', 'AA-t']
    for c in required + ['SS', 'SS-t']:
        mask = df.route_label.eq('T6') if c in ['SS', 'SS-t'] else pd.Series(True, index=df.index)
        if c not in df: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='raise')
        if not np.isfinite(df.loc[mask, c]).all() or (df.loc[mask, c] <= 0).any():
            raise ValueError(f'Missing/nonpositive required process input: {c}')
    for key in phase_descriptors:
        stage, _, source_column = PHASE_DESCRIPTORS[key]
        alias_column(df, key, source_column)
        mask = df.route_label.eq('T6') if stage == 'SS' else pd.Series(True, index=df.index)
        if key not in df: df[key] = np.nan
        df[key] = pd.to_numeric(df[key], errors='raise')
        if not np.isfinite(df.loc[mask, key]).all():
            raise ValueError(f'Missing CALPHAD descriptor {key}; run CALPHAD externally first.')
        if key.startswith('f_') and not df.loc[mask, key].between(0, 1).all():
            raise ValueError(f'{key} must be a fraction in [0,1], not percent.')
    if target:
        if target not in df or not np.isfinite(pd.to_numeric(df[target], errors='coerce')).all():
            raise ValueError(f'Every training row needs a finite {target}.')
    return df

def check_search_bounds(df):
    """Validate candidate coordinates against the configured design bounds."""
    for element in ['Ti','Sc','Ag','Er','V','Y','Ni']:
        if element in df and not pd.to_numeric(df[element], errors='coerce').eq(0).all():
            raise ValueError(f'{element} is outside the design space.')
    for c, (low, high) in BOUNDS.items():
        alias = next((old for old, new in PROCESS_ALIASES.items() if new == c), None)
        col = c if c in df else alias
        if col not in df: raise ValueError(f'Missing search coordinate {c}')
        mask = df.route_label.eq('T6') if c in ['TSS', 'tSS'] else pd.Series(True, index=df.index)
        if not pd.to_numeric(df.loc[mask, col], errors='coerce').between(low, high).all():
            raise ValueError(f'Candidate coordinate {c} must be in [{low}, {high}].')


STAGE_TYPES = ["EXT", "SS", "AA"]
PHASE_TYPES = ["Fcc", "Mg2Si", "Q_Al5Cu2Mg8Si6", "Al15_FeMn3Si2"]
PHASE_ATTRIBUTES = list(PHASE_DESCRIPTORS)
PHASE_PRESENCE_THRESHOLD = 1.0e-6

PHASE_CONSTITUENTS: Mapping[str, Tuple[str, ...]] = {
    "Fcc": tuple(ALL_ELEMENTS),
    "Mg2Si": ("Mg", "Si"),
    "Q_Al5Cu2Mg8Si6": ("Al", "Cu", "Mg", "Si"),
    "Al15_FeMn3Si2": ("Al", "Fe", "Mn", "Si"),
}

EDGE_TYPES: Tuple[Tuple[str, str, str], ...] = (
    ("element", "interacts", "element"),
    ("element", "constituent_of", "phase"),
    ("phase", "has_constituent", "element"),
    ("phase", "belongs_to", "stage"),
    ("stage", "contains", "phase"),
    ("phase", "evolves_to", "phase"),
    ("phase", "evolves_from", "phase"),
    ("stage", "precedes", "stage"),
    ("stage", "follows", "stage"),
)


@dataclass
class PhaseGraphSchema:
    target: str
    element_property_names: List[str]
    element_feature_names: List[str]
    phase_attributes: List[str]
    phase_feature_names: List[str]
    stage_feature_names: List[str]
    phase_threshold: float = PHASE_PRESENCE_THRESHOLD
    ablation: str = "full"


@dataclass
class PhaseRawGraph:
    row_index: int
    sample_no: str
    route: str
    group: str
    element_names: List[str]
    phase_names: List[str]
    phase_stages: List[str]
    stage_names: List[str]
    element_x: np.ndarray
    phase_x: np.ndarray
    stage_x: np.ndarray
    edges: Dict[Tuple[str, str, str], np.ndarray]
    y: Optional[np.ndarray]
    source_group: str = ""
    composition_id: str = ""


@dataclass
class PhasePreprocessor:
    element: MatrixStandardizer
    phase: MatrixStandardizer
    stage: MatrixStandardizer
    target: Optional[TargetStandardizer] = None

    @classmethod
    def fit(
        cls,
        graphs: Sequence[PhaseRawGraph],
        require_target: bool = True,
    ) -> "PhasePreprocessor":
        target_scaler = None
        if require_target:
            targets = [graph.y for graph in graphs if graph.y is not None]
            if len(targets) != len(graphs):
                raise ValueError("Training graphs must all contain a target value.")
            target_scaler = TargetStandardizer().fit(np.asarray(targets, dtype=float))
        return cls(
            element=MatrixStandardizer().fit([graph.element_x for graph in graphs]),
            phase=MatrixStandardizer().fit([graph.phase_x for graph in graphs]),
            stage=MatrixStandardizer().fit([graph.stage_x for graph in graphs]),
            target=target_scaler,
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "element": self.element.state_dict(),
            "phase": self.phase.state_dict(),
            "stage": self.stage.state_dict(),
            "target": None if self.target is None else self.target.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "PhasePreprocessor":
        def matrix(block: Mapping[str, Sequence[float]]) -> MatrixStandardizer:
            return MatrixStandardizer(
                median=np.asarray(block["median"], dtype=float),
                mean=np.asarray(block["mean"], dtype=float),
                scale=np.asarray(block["scale"], dtype=float),
            )

        target_state = state.get("target")
        target = None
        if target_state is not None:
            target = TargetStandardizer(
                mean=np.asarray(target_state["mean"], dtype=float),
                scale=np.asarray(target_state["scale"], dtype=float),
            )
        return cls(
            element=matrix(state["element"]),
            phase=matrix(state["phase"]),
            stage=matrix(state["stage"]),
            target=target,
        )


def make_phase_schema(
    target: str,
    element_property_names: Sequence[str],
    phase_attributes: Optional[Sequence[str]] = None,
    phase_threshold: float = PHASE_PRESENCE_THRESHOLD,
) -> PhaseGraphSchema:
    if target not in PHASE_FEATURES:
        raise ValueError(f"Unknown target: {target}")
    attributes = list(PHASE_FEATURES[target] if phase_attributes is None else phase_attributes)
    unknown = [name for name in attributes if name not in PHASE_ATTRIBUTES]
    if unknown:
        raise ValueError(f"Unknown phase attributes: {unknown}")
    element_feature_names = [
        "wt_fraction",
        "log1p_wt_fraction",
        *list(element_property_names),
        "property_missing_fraction",
    ]
    phase_feature_names = (
        [f"phase_type_{phase}" for phase in PHASE_TYPES]
        + [f"phase_stage_{stage}" for stage in STAGE_TYPES]
        + attributes
        + [f"applicable_{key}" for key in attributes]
    )
    stage_feature_names = (
        [f"stage_type_{stage}" for stage in STAGE_TYPES]
        + [
            "temperature_C",
            "ext_speed_m_min",
            "extrusion_ratio",
            "ss_time_min",
            "aa_time_h",
            "temperature_known",
            "duration_known",
        ]
    )
    return PhaseGraphSchema(
        target=target,
        element_property_names=list(element_property_names),
        element_feature_names=element_feature_names,
        phase_attributes=attributes,
        phase_feature_names=phase_feature_names,
        stage_feature_names=stage_feature_names,
        phase_threshold=float(phase_threshold),
    )


def stage_path(route: str) -> List[str]:
    if route == "T5":
        return ["EXT", "AA"]
    if route == "T6":
        return ["EXT", "SS", "AA"]
    raise ValueError(f"Phase-path HGNN supports T5/T6 only; got {route}.")


def _empty_edges() -> np.ndarray:
    return np.empty((2, 0), dtype=np.int64)


def _edge_array(pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return _empty_edges()
    return np.asarray(pairs, dtype=np.int64).T


def _directed_complete_edges(n_nodes: int) -> np.ndarray:
    pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    return _edge_array(pairs)


def _composition_from_row(row: pd.Series) -> Dict[str, float]:
    solutes = {}
    for element in SOLUTE_ELEMENTS:
        value = numeric(row.get(element, 0.0))
        solutes[element] = 0.0 if not np.isfinite(value) else max(0.0, value)
    solute_sum = sum(solutes.values())
    if solute_sum > 100.0 + 1.0e-6:
        raise ValueError(f"Solute sum exceeds 100 wt.%: {solute_sum:.4f}")
    return {"Al": max(0.0, 100.0 - solute_sum), **solutes}


def _build_element_matrix(
    composition: Mapping[str, float],
    element_properties: pd.DataFrame,
    schema: PhaseGraphSchema,
) -> Tuple[List[str], np.ndarray]:
    active = [element for element in ALL_ELEMENTS if composition.get(element, 0.0) > 0.0]
    rows = []
    for element in active:
        fraction = float(composition[element])
        properties = element_properties.loc[
            element, schema.element_property_names
        ].to_numpy(dtype=float)
        missing_fraction = float(np.mean(~np.isfinite(properties)))
        rows.append(
            np.concatenate(
                [[fraction, np.log1p(fraction)], properties, [missing_fraction]]
            )
        )
    return active, np.asarray(rows, dtype=np.float32)


def _build_stage_matrix(row: pd.Series, names: Sequence[str]) -> np.ndarray:
    rows = []
    for stage in names:
        one_hot = [float(stage == candidate) for candidate in STAGE_TYPES]
        temperature = np.nan
        speed = 0.0
        ratio = 0.0
        ss_minutes = 0.0
        aa_hours = 0.0
        duration_known = 0.0
        if stage == "EXT":
            temperature = numeric(row.get("TEXT"))
            speed = numeric(row.get("EXS(m/min)"))
            ratio = numeric(row.get("EXR"))
        elif stage == "SS":
            temperature = numeric(row.get("SS"))
            ss_minutes = numeric(row.get("SS-t"))
            duration_known = float(np.isfinite(ss_minutes))
        elif stage == "AA":
            temperature = numeric(row.get("AA"))
            aa_hours = numeric(row.get("AA-t"))
            duration_known = float(np.isfinite(aa_hours))
        values = [
            *one_hot,
            temperature,
            0.0 if not np.isfinite(speed) else speed,
            0.0 if not np.isfinite(ratio) else ratio,
            0.0 if not np.isfinite(ss_minutes) else ss_minutes,
            0.0 if not np.isfinite(aa_hours) else aa_hours,
            float(np.isfinite(temperature)),
            duration_known,
        ]
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def _build_phase_matrix(
    row: pd.Series,
    stage_names: Sequence[str],
    schema: PhaseGraphSchema,
) -> Tuple[List[str], List[str], np.ndarray]:
    names: List[str] = []
    stages: List[str] = []
    rows: List[np.ndarray] = []
    for stage in stage_names:
        for phase in PHASE_TYPES:
            applies = [PHASE_DESCRIPTORS[key][:2] == (stage, phase)
                       for key in schema.phase_attributes]
            attributes = [float(row[key]) if active else 0.0
                          for key, active in zip(schema.phase_attributes, applies)]
            vector = np.asarray([
                *[float(phase == candidate) for candidate in PHASE_TYPES],
                *[float(stage == candidate) for candidate in STAGE_TYPES],
                *attributes, *map(float, applies)], dtype=np.float32)
            names.append(phase)
            stages.append(stage)
            rows.append(vector)
    if not rows:
        raise ValueError("No dynamic phase nodes were generated for this sample.")
    return names, stages, np.vstack(rows)


def _build_edges(
    element_names: Sequence[str],
    phase_names: Sequence[str],
    phase_stages: Sequence[str],
    stage_names: Sequence[str],
) -> Dict[Tuple[str, str, str], np.ndarray]:
    edges = {relation: _empty_edges() for relation in EDGE_TYPES}
    edges[("element", "interacts", "element")] = _directed_complete_edges(
        len(element_names)
    )

    constituent_pairs = []
    for element_i, element in enumerate(element_names):
        for phase_i, phase in enumerate(phase_names):
            if element in PHASE_CONSTITUENTS[phase]:
                constituent_pairs.append((element_i, phase_i))
    forward = _edge_array(constituent_pairs)
    edges[("element", "constituent_of", "phase")] = forward
    edges[("phase", "has_constituent", "element")] = forward[[1, 0], :]

    belongs_pairs = [
        (phase_i, stage_names.index(stage))
        for phase_i, stage in enumerate(phase_stages)
    ]
    forward = _edge_array(belongs_pairs)
    edges[("phase", "belongs_to", "stage")] = forward
    edges[("stage", "contains", "phase")] = forward[[1, 0], :]

    sequence_pairs = [(i, i + 1) for i in range(len(stage_names) - 1)]
    forward = _edge_array(sequence_pairs)
    edges[("stage", "precedes", "stage")] = forward
    edges[("stage", "follows", "stage")] = forward[[1, 0], :]

    phase_lookup = {
        (stage, phase): index
        for index, (stage, phase) in enumerate(zip(phase_stages, phase_names))
    }
    evolution_pairs = []
    for stage_a, stage_b in zip(stage_names[:-1], stage_names[1:]):
        for phase in PHASE_TYPES:
            key_a, key_b = (stage_a, phase), (stage_b, phase)
            if key_a in phase_lookup and key_b in phase_lookup:
                evolution_pairs.append((phase_lookup[key_a], phase_lookup[key_b]))
    forward = _edge_array(evolution_pairs)
    edges[("phase", "evolves_to", "phase")] = forward
    edges[("phase", "evolves_from", "phase")] = forward[[1, 0], :]
    return edges


def build_phase_raw_graphs(
    df: pd.DataFrame,
    element_properties: pd.DataFrame,
    schema: PhaseGraphSchema,
    require_target: bool = True,
) -> List[PhaseRawGraph]:
    graphs: List[PhaseRawGraph] = []
    for row_index, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Building {schema.target} phase-path graphs"
    ):
        route = infer_route(row)
        if route not in {"T5", "T6"}:
            continue
        composition = _composition_from_row(row)
        element_names, element_x = _build_element_matrix(
            composition, element_properties, schema
        )
        stages = stage_path(route)
        stage_x = _build_stage_matrix(row, stages)
        phase_names, phase_stages, phase_x = _build_phase_matrix(row, stages, schema)
        edges = _build_edges(element_names, phase_names, phase_stages, stages)
        target_value = numeric(row.get(schema.target, np.nan))
        if require_target and not np.isfinite(target_value):
            raise ValueError(f"Row {row_index} is missing target {schema.target}.")
        y = (
            np.asarray([target_value], dtype=np.float32)
            if np.isfinite(target_value)
            else None
        )
        sample_no = str(row.get("sample_no", row.get("sample_id", row_index))).strip()
        graphs.append(
            PhaseRawGraph(
                row_index=int(row_index),
                sample_no=sample_no,
                route=route,
                group=make_composition_group(row),
                element_names=element_names,
                phase_names=phase_names,
                phase_stages=phase_stages,
                stage_names=stages,
                element_x=element_x,
                phase_x=phase_x,
                stage_x=stage_x,
                edges=edges,
                y=y,
                source_group=str(row.get("source_id", "")).strip(),
                composition_id=str(row.get("composition_id", "")).strip(),
            )
        )
    if not graphs:
        raise ValueError("No T5/T6 phase-path graphs were generated.")
    for graph in graphs:
        apply_ablation(graph, schema)
    return graphs


def apply_ablation(graph, schema):
    """Keep node identities/topology fixed when removing descriptor families."""
    if schema.ablation not in {"full", "base", "no_element", "no_phase", "no_evolution"}:
        raise ValueError(f"Unknown ablation: {schema.ablation}")
    if schema.ablation in {"base", "no_element"}:
        graph.element_x[:, 2:] = 0
    if schema.ablation in {"base", "no_phase"}:
        graph.phase_x[:, len(PHASE_TYPES) + len(STAGE_TYPES):] = 0
    if schema.ablation == "no_evolution":
        for key in [("phase", "evolves_to", "phase"), ("phase", "evolves_from", "phase")]:
            graph.edges[key] = _empty_edges()


def prepare_phase_dataset(
    data_file: Path,
    element_property_file: Path,
    target: str,
    selected_element_properties: Sequence[str],
    phase_attributes: Optional[Sequence[str]] = None,
    phase_threshold: float = PHASE_PRESENCE_THRESHOLD,
    require_target: bool = True,
    ablation: str = "full",
) -> Tuple[pd.DataFrame, pd.DataFrame, PhaseGraphSchema, List[PhaseRawGraph]]:
    selected = PHASE_FEATURES[target] if phase_attributes is None else list(phase_attributes)
    df = normalize_inputs(read_table(data_file), SOLUTE_ELEMENTS, selected,
                          target if require_target else None)

    required_elements = ["Al"]
    for element in SOLUTE_ELEMENTS:
        values = pd.to_numeric(df.get(element, pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
        if (values > 0).any():
            required_elements.append(element)
    element_properties, property_names = load_element_properties(
        element_property_file,
        required_elements=required_elements,
        selected_property_names=selected_element_properties,
    )
    schema = make_phase_schema(
        target,
        property_names,
        phase_attributes=selected,
        phase_threshold=phase_threshold,
    )
    schema.ablation = ablation
    graphs = build_phase_raw_graphs(
        df, element_properties, schema, require_target=require_target
    )
    return df, element_properties, schema, graphs


def to_heterodata(
    graph: PhaseRawGraph,
    preprocessor: PhasePreprocessor,
    graph_index: int,
) -> HeteroData:
    require_torch_geometric()
    data = HeteroData()
    data["element"].x = torch.tensor(
        preprocessor.element.transform(graph.element_x), dtype=torch.float32
    )
    data["phase"].x = torch.tensor(
        preprocessor.phase.transform(graph.phase_x), dtype=torch.float32
    )
    data["stage"].x = torch.tensor(
        preprocessor.stage.transform(graph.stage_x), dtype=torch.float32
    )
    data["phase"].phase_type_index = torch.tensor(
        [PHASE_TYPES.index(value) for value in graph.phase_names], dtype=torch.long
    )
    data["phase"].stage_type_index = torch.tensor(
        [STAGE_TYPES.index(value) for value in graph.phase_stages], dtype=torch.long
    )
    data["stage"].stage_type_index = torch.tensor(
        [STAGE_TYPES.index(value) for value in graph.stage_names], dtype=torch.long
    )
    for relation in EDGE_TYPES:
        data[relation].edge_index = torch.tensor(
            graph.edges[relation], dtype=torch.long
        )
    if graph.y is not None:
        if preprocessor.target is None:
            raise RuntimeError("Target scaler is unavailable for a labelled graph.")
        scaled = preprocessor.target.transform(graph.y.reshape(1, -1))
        data.y = torch.tensor(scaled, dtype=torch.float32)
    data.graph_index = torch.tensor([graph_index], dtype=torch.long)
    return data


class ElementPhaseStageHGNN(nn.Module):
    def __init__(
        self,
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],
        element_dim: int,
        phase_dim: int,
        stage_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        require_torch_geometric()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads.")
        self.dropout = dropout
        self.encoders = nn.ModuleDict(
            {
                "element": nn.Sequential(nn.Linear(element_dim, hidden_dim), nn.ELU()),
                "phase": nn.Sequential(nn.Linear(phase_dim, hidden_dim), nn.ELU()),
                "stage": nn.Sequential(nn.Linear(stage_dim, hidden_dim), nn.ELU()),
            }
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            relation_convs = {
                relation: GATv2Conv(
                    (-1, -1),
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for relation in metadata[1]
            }
            self.convs.append(HeteroConv(relation_convs, aggr="sum"))
            self.norms.append(
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim)
                        for node_type in ("element", "phase", "stage")
                    }
                )
            )
        graph_dim = hidden_dim * 6
        self.head = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_graph(self, batch: HeteroData) -> torch.Tensor:
        x_dict = {
            node_type: self.encoders[node_type](values)
            for node_type, values in batch.x_dict.items()
        }
        for conv, norms in zip(self.convs, self.norms):
            updated = conv(x_dict, batch.edge_index_dict)
            next_x = {}
            for node_type, values in x_dict.items():
                message = updated.get(node_type, torch.zeros_like(values))
                next_x[node_type] = norms[node_type](
                    values
                    + F.dropout(F.elu(message), self.dropout, training=self.training)
                )
            x_dict = next_x
        summaries = []
        for node_type in ("element", "phase", "stage"):
            node_batch = batch[node_type].batch
            summaries.extend(
                [
                    global_mean_pool(x_dict[node_type], node_batch),
                    global_max_pool(x_dict[node_type], node_batch),
                ]
            )
        return torch.cat(summaries, dim=-1)

    def forward(self, batch: HeteroData) -> torch.Tensor:
        return self.head(self.encode_graph(batch))


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    with_truth: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    model.eval()
    predictions, truths, indices = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            predictions.append(model(batch).cpu().numpy())
            if with_truth:
                truths.append(batch.y.cpu().numpy())
            indices.append(batch.graph_index.cpu().numpy())
    truth = np.vstack(truths) if with_truth else None
    return np.vstack(predictions), truth, np.concatenate(indices)


def _make_model(data: HeteroData, config: TrainConfig) -> ElementPhaseStageHGNN:
    return ElementPhaseStageHGNN(
        metadata=data.metadata(),
        element_dim=data["element"].x.shape[1],
        phase_dim=data["phase"].x.shape[1],
        stage_dim=data["stage"].x.shape[1],
        hidden_dim=config.hidden_dim,
        heads=config.heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    )


def train_with_early_stopping(
    train_data: Sequence[HeteroData],
    validation_data: Sequence[HeteroData],
    config: TrainConfig,
    device: torch.device,
    label: str,
) -> Tuple[ElementPhaseStageHGNN, pd.DataFrame, int]:
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = DataLoader(
        validation_data, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    model = _make_model(train_data[0], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_loss = math.inf
    best_state = None
    best_epoch = 1
    stale = 0
    rows = []
    bar = tqdm(range(1, config.max_epochs + 1), desc=label, leave=False)
    for epoch in bar:
        model.train()
        train_sum = 0.0
        count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = F.mse_loss(prediction, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_sum += float(loss.item()) * batch.y.shape[0]
            count += batch.y.shape[0]
        validation_prediction, validation_truth, _ = _evaluate(
            model, validation_loader, device
        )
        validation_loss = float(
            np.mean((validation_prediction - validation_truth) ** 2)
        )
        train_loss = train_sum / max(1, count)
        rows.append(
            {
                "epoch": epoch,
                "train_loss_standardized": train_loss,
                "validation_loss_standardized": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        bar.set_postfix(train=f"{train_loss:.3f}", val=f"{validation_loss:.3f}")
        if validation_loss < best_loss - config.min_delta:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows), best_epoch


def fit_fixed_epochs(
    data: Sequence[HeteroData],
    config: TrainConfig,
    device: torch.device,
    epochs: int,
    seed: int,
    label: str,
) -> ElementPhaseStageHGNN:
    set_seed(seed)
    loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    model = _make_model(data[0], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in tqdm(range(max(1, epochs)), desc=label, leave=False):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


def _checkpoint_payload(
    model: ElementPhaseStageHGNN,
    preprocessor: PhasePreprocessor,
    schema: PhaseGraphSchema,
    config: TrainConfig,
    data: HeteroData,
    selected_epoch: int,
    split: str,
) -> Dict[str, object]:
    return {
        "format_version": 2,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metadata": data.metadata(),
        "element_dim": data["element"].x.shape[1],
        "phase_dim": data["phase"].x.shape[1],
        "stage_dim": data["stage"].x.shape[1],
        "schema": asdict(schema),
        "preprocessor": preprocessor.state_dict(),
        "config": asdict(config),
        "selected_epoch": int(selected_epoch),
        "split": split,
    }


def run_random_cv(
    graphs: Sequence[PhaseRawGraph],
    schema: PhaseGraphSchema,
    output_dir: Path,
    config: TrainConfig,
    device: Optional[torch.device] = None,
    save_checkpoints: bool = True,
    split_strategy: str = "random",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_torch_geometric()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if split_strategy not in {"random", "group"}:
        raise ValueError("split_strategy must be random or group")
    groups = leakage_groups(graphs)
    if split_strategy == "group":
        validate_group_metadata(graphs)
        if len(set(groups)) < config.n_splits:
            raise ValueError("Too few independent source/composition groups for the requested folds.")
        splitter = GroupKFold(n_splits=config.n_splits)
        splits = splitter.split(np.arange(len(graphs)), groups=groups)
    else:
        splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        splits = splitter.split(np.arange(len(graphs)))
    predictions = np.full(len(graphs), np.nan, dtype=float)
    fold_assignment = np.full(len(graphs), -1, dtype=int)
    fold_rows = []
    histories = []
    for fold, (outer_train, test_idx) in enumerate(
        tqdm(
            splits,
            total=config.n_splits,
            desc=f"{schema.target} {split_strategy} CV",
        ),
        start=1,
    ):
        if split_strategy == "group":
            inner = GroupShuffleSplit(n_splits=1, test_size=config.validation_fraction,
                                      random_state=config.seed + fold)
            train_rel, val_rel = next(inner.split(outer_train, groups=groups[outer_train]))
        else:
            inner = ShuffleSplit(n_splits=1, test_size=config.validation_fraction,
                                 random_state=config.seed + fold)
            train_rel, val_rel = next(inner.split(outer_train))
        train_idx = outer_train[train_rel]
        val_idx = outer_train[val_rel]
        preprocessor = PhasePreprocessor.fit([graphs[i] for i in train_idx])
        train_data = [
            to_heterodata(graphs[i], preprocessor, graph_index=i) for i in train_idx
        ]
        validation_data = [
            to_heterodata(graphs[i], preprocessor, graph_index=i) for i in val_idx
        ]
        set_seed(config.seed + fold)
        _, history, best_epoch = train_with_early_stopping(
            train_data,
            validation_data,
            config,
            device,
            label=f"{schema.target} fold {fold}",
        )
        history.insert(0, "fold", fold)
        histories.append(history)

        preprocessor = PhasePreprocessor.fit([graphs[i] for i in outer_train])
        outer_data = [
            to_heterodata(graphs[i], preprocessor, graph_index=i)
            for i in outer_train
        ]
        model = fit_fixed_epochs(
            outer_data,
            config,
            device,
            best_epoch,
            seed=config.seed + 10_000 + fold,
            label=f"{schema.target} fold {fold} refit",
        )
        test_data = [
            to_heterodata(graphs[i], preprocessor, graph_index=i) for i in test_idx
        ]
        loader = DataLoader(
            test_data, batch_size=config.batch_size, shuffle=False, num_workers=0
        )
        pred_scaled, true_scaled, graph_indices = _evaluate(model, loader, device)
        prediction = preprocessor.target.inverse_transform(pred_scaled).ravel()
        truth = preprocessor.target.inverse_transform(true_scaled).ravel()
        predictions[graph_indices] = prediction
        fold_assignment[graph_indices] = fold
        metrics = regression_metrics(truth, prediction)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "n_refit_train": len(outer_train),
                "n_test": len(test_idx),
                "selected_epoch": best_epoch,
                **metrics,
            }
        )
        if save_checkpoints:
            torch.save(
                _checkpoint_payload(
                    model,
                    preprocessor,
                    schema,
                    config,
                    outer_data[0],
                    best_epoch,
                    split=f"{split_strategy}_fold_{fold}",
                ),
                output_dir / f"fold_{fold:02d}.pt",
            )
    if np.isnan(predictions).any():
        raise RuntimeError("OOF prediction coverage is incomplete.")
    oof_rows = []
    for index, graph in enumerate(graphs):
        oof_rows.append(
            {
                "row_index": graph.row_index,
                "sample_no": graph.sample_no,
                "route": graph.route,
                "composition_group": graph.group,
                "composition_id": graph.composition_id,
                "source_id": graph.source_group,
                "fold": fold_assignment[index],
                f"true_{schema.target}": float(graph.y[0]),
                f"pred_{schema.target}": predictions[index],
                f"error_{schema.target}": predictions[index] - float(graph.y[0]),
            }
        )
    oof = pd.DataFrame(oof_rows)
    summary_rows = []
    for scope, subset in [("all", oof)] + list(oof.groupby("route")):
        summary_rows.append(
            {
                "scope": scope,
                "target": schema.target,
                "n": len(subset),
                **regression_metrics(
                    subset[f"true_{schema.target}"],
                    subset[f"pred_{schema.target}"],
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)
    history = pd.concat(histories, ignore_index=True)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "metrics_overall_and_by_route.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(output_dir / "metrics_by_fold.csv", index=False, encoding="utf-8-sig")
    history.to_csv(output_dir / "learning_curves.csv", index=False, encoding="utf-8-sig")
    calibration = calibrate_lcb(oof, schema.target)
    (output_dir / "lcb_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": asdict(schema),
                "config": asdict(config),
                "device": str(device),
                "validation": split_strategy,
                "feature_set": "selected_descriptors",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return oof, summary, folds


def validate_group_metadata(graphs):
    """Check source identifiers used by grouped validation."""
    broad = {"literature", "paper", "publication", "factory", "industrial",
             "laboratory", "lab", "experiment", "文献", "工厂", "工业", "实验室"}
    missing = [g.sample_no for g in graphs if not g.source_group.strip()
               or g.source_group.strip().lower() in {"nan", "none", "null"}]
    generic = [g.sample_no for g in graphs if g.source_group.strip().casefold() in broad]
    if missing or generic:
        raise ValueError(
            f"Grouped CV requires a specific source_id for each record "
            f"(paper, melt, or production batch); missing={len(missing)}, "
            f"generic={len(generic)}. Example sample IDs: {(missing + generic)[:5]}"
        )


def leakage_groups(graphs):
    """Connected components of shared chemistry, composition ID, or source ID."""
    parent = list(range(len(graphs)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, graph in enumerate(graphs):
        keys = [("composition", graph.group)]
        composition_id = getattr(graph, "composition_id", "").strip()
        if composition_id and composition_id.lower() not in {"nan", "none", "null"}:
            keys.append(("composition_id", composition_id))
        if graph.source_group and graph.source_group.lower() != "nan":
            keys.append(("source", graph.source_group))
        for key in keys:
            if key in seen: parent[root(i)] = root(seen[key])
            else: seen[key] = i
    return np.array([root(i) for i in range(len(graphs))])


def calibrate_lcb(oof: pd.DataFrame, target: str) -> Dict[str, object]:
    residual = (oof[f"true_{target}"] - oof[f"pred_{target}"]).to_numpy(dtype=float)
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Calibration needs nonempty finite OOF residuals.")
    return {
        "target": target, "n_oof": len(residual),
        "residual_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "kappa": 1.0,
        "formula": "sigma_cal = sqrt(ensemble_std^2 + residual_rmse^2); LCB = mean - sigma_cal",
        "method": "ensemble_spread_and_oof_rmse",
    }


def train_full_ensemble(
    graphs: Sequence[PhaseRawGraph],
    schema: PhaseGraphSchema,
    output_dir: Path,
    config: TrainConfig,
    n_members: int = 5,
    device: Optional[torch.device] = None,
    split_strategy: str = "group",
) -> pd.DataFrame:
    require_torch_geometric()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    indices = np.arange(len(graphs))
    if split_strategy not in {"random", "group"}:
        raise ValueError("split_strategy must be random or group")
    if split_strategy == "group":
        validate_group_metadata(graphs)
        groups = leakage_groups(graphs)
        if len(set(groups)) < 2:
            raise ValueError("At least two independent groups are required for ensemble epoch selection.")
    for member in range(1, n_members + 1):
        splitter_class = GroupShuffleSplit if split_strategy == "group" else ShuffleSplit
        splitter = splitter_class(
            n_splits=1,
            test_size=config.validation_fraction,
            random_state=config.seed + 100 * member,
        )
        train_idx, val_idx = next(splitter.split(indices, groups=groups if split_strategy == "group" else None))
        provisional = PhasePreprocessor.fit([graphs[i] for i in train_idx])
        train_data = [
            to_heterodata(graphs[i], provisional, graph_index=i) for i in train_idx
        ]
        val_data = [
            to_heterodata(graphs[i], provisional, graph_index=i) for i in val_idx
        ]
        set_seed(config.seed + member)
        _, _, best_epoch = train_with_early_stopping(
            train_data,
            val_data,
            config,
            device,
            label=f"{schema.target} ensemble {member}",
        )
        final_preprocessor = PhasePreprocessor.fit(graphs)
        full_data = [
            to_heterodata(graph, final_preprocessor, graph_index=i)
            for i, graph in enumerate(graphs)
        ]
        model = fit_fixed_epochs(
            full_data,
            config,
            device,
            best_epoch,
            seed=config.seed + 20_000 + member,
            label=f"{schema.target} ensemble {member} refit",
        )
        checkpoint = _checkpoint_payload(
            model,
            final_preprocessor,
            schema,
            config,
            full_data[0],
            best_epoch,
            split="full_data_ensemble",
        )
        checkpoint["ensemble_member"] = member
        path = output_dir / f"ensemble_{member:02d}.pt"
        torch.save(checkpoint, path)
        rows.append(
            {"member": member, "selected_epoch": best_epoch, "checkpoint": str(path)}
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "ensemble_manifest.csv", index=False, encoding="utf-8-sig")
    return table


def load_phase_checkpoint(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> Tuple[ElementPhaseStageHGNN, PhasePreprocessor, PhaseGraphSchema, TrainConfig]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("Unsupported checkpoint format; retrain with the current schema.")
    schema = PhaseGraphSchema(**checkpoint["schema"])
    config = TrainConfig(**checkpoint["config"])
    model = ElementPhaseStageHGNN(
        metadata=checkpoint["metadata"],
        element_dim=checkpoint["element_dim"],
        phase_dim=checkpoint["phase_dim"],
        stage_dim=checkpoint["stage_dim"],
        hidden_dim=config.hidden_dim,
        heads=config.heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preprocessor = PhasePreprocessor.from_state_dict(checkpoint["preprocessor"])
    return model, preprocessor, schema, config


def predict_checkpoint(
    checkpoint_path: Path,
    graphs: Sequence[PhaseRawGraph],
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocessor, _, _ = load_phase_checkpoint(checkpoint_path, device)
    data = [
        to_heterodata(graph, preprocessor, graph_index=i)
        for i, graph in enumerate(graphs)
    ]
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    scaled, _, indices = _evaluate(model, loader, device, with_truth=False)
    prediction = preprocessor.target.inverse_transform(scaled).ravel()
    ordered = np.empty(len(graphs), dtype=float)
    ordered[indices] = prediction
    return ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--target', choices=['all','UTS','EL'], default='all')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--element-properties', type=Path, required=True)
    p.add_argument('--split', choices=['random','group'], default='group')
    p.add_argument('--output', type=Path, default=Path('outputs'))
    p.add_argument('--ensemble-members', type=int, default=5)
    p.add_argument('--ablation', choices=['full','base','no_element','no_phase','no_evolution'], default='full')
    p.add_argument('--threads', type=int, default=1)
    a = p.parse_args()
    if a.ensemble_members < 2: p.error('Use at least two ensemble members.')
    torch.set_num_threads(a.threads)
    if a.epochs < 1 or a.folds < 2: p.error('Use positive epochs and at least two folds.')
    cfg = TrainConfig(max_epochs=a.epochs, n_splits=a.folds)
    for target in ['UTS','EL'] if a.target == 'all' else [a.target]:
        out = a.output / a.split / 'full' / target
        if a.ablation != 'full': out = out / a.ablation
        out.mkdir(parents=True, exist_ok=True)
        selected = ELEMENT_FEATURES[target]
        _, _, schema, graphs = prepare_phase_dataset(a.data, a.element_properties, target,
            selected, phase_attributes=PHASE_FEATURES[target], ablation=a.ablation)
        if a.split == 'group': validate_group_metadata(graphs)
        run_random_cv(graphs, schema, out/'cross_validation', cfg, split_strategy=a.split)
        train_full_ensemble(graphs, schema, out/'ensemble', cfg, n_members=a.ensemble_members,
                            split_strategy=a.split)
        print(f'Local results: {out}')

if __name__ == '__main__': main()
