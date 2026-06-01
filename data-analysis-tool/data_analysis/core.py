"""
Data Analysis Toolkit — core module
====================================
Contains the **DataInspector** class for end-to-end data ingestion,
cleaning, feature-engineering, and visualisation, and the
**PlottingMethods** helper class for granular chart generation.
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    LabelEncoder,
    OrdinalEncoder,
)

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
#                         DataInspector
# ═══════════════════════════════════════════════════════════════════
class DataInspector:
    """
    A reusable class for CSV data ingestion, advanced cleaning,
    feature engineering preparation, and high-level statistical
    visualisation inside Google Colab.
    """

    # Strings that should be treated as missing values
    _GARBAGE_STRINGS = [
        "?", "n/a", "N/A", "na", "NA", "null", "NULL",
        "none", "None", "NONE", " ", "", "  ", "---",
        "-", ".", "undefined", "NaN", "nan",
    ]

    def __init__(self):
        """Initialise DataInspector with an empty DataFrame."""
        self.df: pd.DataFrame | None = None
        self._normalized_numeric: pd.DataFrame | None = None
        self._normalized_categorical: pd.DataFrame | None = None

    # ───────────────────── 1. Data Ingestion ─────────────────────

    def upload_data(self):
        """
        Upload a CSV file interactively in Google Colab.
        Automatically sanitises garbage strings and attempts
        auto-type correction on every column.
        """
        try:
            from google.colab import files as colab_files

            uploaded = colab_files.upload()
            if not uploaded:
                print("⚠  No file uploaded.")
                return

            filename = list(uploaded.keys())[0]
            self.df = pd.read_csv(
                filename,
                na_values=self._GARBAGE_STRINGS,
                keep_default_na=True,
            )
        except ImportError:
            # Fallback for non-Colab environments
            path = input("Enter CSV file path: ").strip()
            self.df = pd.read_csv(
                path,
                na_values=self._GARBAGE_STRINGS,
                keep_default_na=True,
            )

        self._sanitise()
        print(f"✅  Loaded {self.df.shape[0]} rows × {self.df.shape[1]} columns.")

    # ── internal helpers ──

    def _sanitise(self):
        """Replace remaining garbage strings and auto-convert types."""
        if self.df is None:
            return

        # Replace any leftover garbage strings that slipped through
        self.df.replace(
            {s: np.nan for s in self._GARBAGE_STRINGS}, inplace=True
        )

        # Auto-type correction: try to coerce object columns to numeric
        for col in self.df.columns:
            if self.df[col].dtype == object:
                converted = pd.to_numeric(self.df[col], errors="coerce")
                # Only keep conversion if it didn't turn everything into NaN
                if converted.notna().sum() > 0:
                    self.df[col] = converted

    def _check_data(self):
        """Guard clause — prints a warning if no data is loaded."""
        if self.df is None or self.df.empty:
            print("⚠  No data loaded. Call upload_data() first.")
            return False
        return True

    def _numeric_cols(self):
        """Return list of numeric column names."""
        return self.df.select_dtypes(include="number").columns.tolist()

    def _categorical_cols(self):
        """Return list of categorical (object / category) column names."""
        return self.df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    # ───────────────── 2. Structural Analysis ────────────────────

    def data_summary(self):
        """
        Display row/column counts, first 20 rows, and a
        breakdown of numerical vs categorical columns.
        """
        if not self._check_data():
            return

        print("=" * 60)
        print(f"  Rows   : {self.df.shape[0]}")
        print(f"  Columns: {self.df.shape[1]}")
        print("=" * 60)

        num_cols = self._numeric_cols()
        cat_cols = self._categorical_cols()
        print(f"\n  Numerical columns   ({len(num_cols)}): {num_cols}")
        print(f"  Categorical columns ({len(cat_cols)}): {cat_cols}")

        print("\n── First 20 rows ──")
        try:
            from IPython.display import display
            display(self.df.head(20))
        except ImportError:
            print(self.df.head(20).to_string())

    def column_details(self):
        """Print dtype, non-null count and unique count for each column."""
        if not self._check_data():
            return
        info_df = pd.DataFrame(
            {
                "dtype": self.df.dtypes,
                "non_null": self.df.notna().sum(),
                "null": self.df.isna().sum(),
                "unique": self.df.nunique(),
            }
        )
        try:
            from IPython.display import display
            display(info_df)
        except ImportError:
            print(info_df.to_string())

    def get_categorical_summary(self):
        """Show value counts for every categorical column."""
        if not self._check_data():
            return
        for col in self._categorical_cols():
            print(f"\n── {col} ──")
            print(self.df[col].value_counts())

    def show_missing_data(self):
        """Display missing value counts and percentages per column."""
        if not self._check_data():
            return
        total = self.df.isna().sum()
        pct = (total / len(self.df) * 100).round(2)
        missing = pd.DataFrame({"missing_count": total, "missing_%": pct})
        missing = missing[missing["missing_count"] > 0].sort_values(
            "missing_%", ascending=False
        )
        if missing.empty:
            print("✅  No missing values found.")
        else:
            try:
                from IPython.display import display
                display(missing)
            except ImportError:
                print(missing.to_string())

    # ───────────────── Intelligent Imputation ────────────────────

    def handle_missing_values(
        self, strategy: str = "mean", constant_value=0
    ):
        """
        Fill missing values using the chosen strategy.

        Parameters
        ----------
        strategy : str
            One of ``'mean'``, ``'median'``, ``'mode'``, or ``'constant'``.
        constant_value
            Value used when ``strategy='constant'``.
        """
        if not self._check_data():
            return

        num_cols = self._numeric_cols()
        cat_cols = self._categorical_cols()

        if strategy == "mean":
            self.df[num_cols] = self.df[num_cols].fillna(
                self.df[num_cols].mean()
            )
            for col in cat_cols:
                self.df[col].fillna(self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "Unknown", inplace=True)

        elif strategy == "median":
            self.df[num_cols] = self.df[num_cols].fillna(
                self.df[num_cols].median()
            )
            for col in cat_cols:
                self.df[col].fillna(self.df[col].mode().iloc[0] if not self.df[col].mode().empty else "Unknown", inplace=True)

        elif strategy == "mode":
            for col in self.df.columns:
                if not self.df[col].mode().empty:
                    self.df[col].fillna(
                        self.df[col].mode().iloc[0], inplace=True
                    )

        elif strategy == "constant":
            self.df.fillna(constant_value, inplace=True)

        else:
            print(f"⚠  Unknown strategy '{strategy}'. Use mean/median/mode/constant.")
            return

        print(f"✅  Missing values handled using '{strategy}' strategy.")

    # ───────────── Duplicate & Outlier Management ────────────────

    def remove_duplicates(self):
        """Remove exact duplicate rows."""
        if not self._check_data():
            return
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        print(f"✅  Removed {removed} duplicate rows. ({len(self.df)} remaining)")

    def handle_outliers(
        self,
        columns: list | None = None,
        find_and_delete: bool = False,
        threshold: float = 1.5,
    ):
        """
        IQR-based outlier detection. Flags or removes rows.

        Parameters
        ----------
        columns : list, optional
            Numeric columns to check. Defaults to all numeric columns.
        find_and_delete : bool
            If True, delete outlier rows; otherwise just flag them.
        threshold : float
            IQR multiplier (default 1.5).
        """
        if not self._check_data():
            return

        if columns is None:
            columns = self._numeric_cols()

        outlier_mask = pd.Series(False, index=self.df.index)

        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(
                self.df[col]
            ):
                continue
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            col_outliers = (self.df[col] < lower) | (self.df[col] > upper)
            n = col_outliers.sum()
            print(f"  {col}: {n} outliers detected  [range {lower:.2f} – {upper:.2f}]")
            outlier_mask |= col_outliers

        total = outlier_mask.sum()
        if find_and_delete:
            self.df = self.df[~outlier_mask].reset_index(drop=True)
            print(f"✅  Deleted {total} outlier rows. ({len(self.df)} remaining)")
        else:
            self.df["_outlier_flag"] = outlier_mask
            print(f"✅  Flagged {total} rows in '_outlier_flag' column.")

    # ───────────────── Targeted Deletion ─────────────────────────

    def delete_rows(self):
        """
        Interactively delete rows by entering comma-separated indices.
        """
        if not self._check_data():
            return
        raw = input("Enter row indices to delete (comma-separated): ")
        try:
            indices = [int(i.strip()) for i in raw.split(",") if i.strip()]
            self.df.drop(index=indices, inplace=True, errors="ignore")
            self.df.reset_index(drop=True, inplace=True)
            print(f"✅  Deleted rows. {len(self.df)} rows remaining.")
        except ValueError:
            print("⚠  Invalid input. Please provide comma-separated integers.")

    def delete_columns(self):
        """
        Interactively delete columns by entering comma-separated names.
        """
        if not self._check_data():
            return
        raw = input("Enter column names to delete (comma-separated): ")
        cols = [c.strip() for c in raw.split(",") if c.strip()]
        self.df.drop(columns=cols, inplace=True, errors="ignore")
        print(f"✅  Remaining columns: {list(self.df.columns)}")

    # ═══════════════ 3. Feature Engineering ═════════════════════

    def extract_normalized_numeric_data(
        self, method: str = "minmax"
    ) -> pd.DataFrame:
        """
        Scale numeric columns.

        Parameters
        ----------
        method : str
            ``'minmax'``, ``'standard'`` (Z-score), or ``'robust'`` (IQR).

        Returns
        -------
        pd.DataFrame
            Scaled numeric data.
        """
        if not self._check_data():
            return pd.DataFrame()

        num_data = self.df[self._numeric_cols()].dropna(axis=1, how="all")
        if num_data.empty:
            print("⚠  No numeric columns to normalise.")
            return pd.DataFrame()

        # Fill remaining NaNs with column median before scaling
        num_data = num_data.fillna(num_data.median())

        scalers = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }
        scaler = scalers.get(method)
        if scaler is None:
            print(f"⚠  Unknown method '{method}'. Use minmax/standard/robust.")
            return pd.DataFrame()

        scaled = pd.DataFrame(
            scaler.fit_transform(num_data),
            columns=num_data.columns,
            index=num_data.index,
        )
        self._normalized_numeric = scaled
        print(f"✅  Numeric data scaled using '{method}'.")
        return scaled

    def extract_normalized_categorical_data(
        self, method: str = "onehot"
    ) -> pd.DataFrame:
        """
        Encode categorical columns.

        Parameters
        ----------
        method : str
            ``'onehot'``, ``'ordinal'``, or ``'uniform'`` (ordinal scaled 0-1).

        Returns
        -------
        pd.DataFrame
            Encoded categorical data.
        """
        if not self._check_data():
            return pd.DataFrame()

        cat_cols = self._categorical_cols()
        if not cat_cols:
            print("⚠  No categorical columns to encode.")
            return pd.DataFrame()

        cat_data = self.df[cat_cols].fillna("Unknown")

        if method == "onehot":
            encoded = pd.get_dummies(cat_data, dtype=int)

        elif method == "ordinal":
            enc = OrdinalEncoder()
            encoded = pd.DataFrame(
                enc.fit_transform(cat_data),
                columns=cat_data.columns,
                index=cat_data.index,
            )

        elif method == "uniform":
            enc = OrdinalEncoder()
            ordinal = enc.fit_transform(cat_data)
            # Scale each column 0-1
            ordinal_df = pd.DataFrame(
                ordinal, columns=cat_data.columns, index=cat_data.index
            )
            for col in ordinal_df.columns:
                col_max = ordinal_df[col].max()
                if col_max > 0:
                    ordinal_df[col] = ordinal_df[col] / col_max
            encoded = ordinal_df

        else:
            print(f"⚠  Unknown method '{method}'. Use onehot/ordinal/uniform.")
            return pd.DataFrame()

        self._normalized_categorical = encoded
        print(f"✅  Categorical data encoded using '{method}'.")
        return encoded

    def create_normalized_data_df(
        self,
        numeric_method: str = "minmax",
        categorical_method: str = "onehot",
    ) -> pd.DataFrame:
        """
        Create a unified DataFrame with scaled numeric data and
        encoded categorical data merged together.
        """
        if self._normalized_numeric is None:
            self.extract_normalized_numeric_data(method=numeric_method)
        if self._normalized_categorical is None:
            self.extract_normalized_categorical_data(method=categorical_method)

        parts = [
            p
            for p in (self._normalized_numeric, self._normalized_categorical)
            if p is not None and not p.empty
        ]
        if not parts:
            print("⚠  Nothing to merge.")
            return pd.DataFrame()

        merged = pd.concat(parts, axis=1)
        print(f"✅  Merged DataFrame: {merged.shape[0]} rows × {merged.shape[1]} columns.")
        return merged

    # ═══════════════ 4. Interactive Visualisation ════════════════

    def plot_numerical(self, column_names: list | None = None):
        """
        For each numeric column, show a 3-panel subplot:
        Horizontal Violin/Box | Scatter (Index vs Value) | Histogram.
        """
        if not self._check_data():
            return
        cols = column_names or self._numeric_cols()
        cols = [c for c in cols if c in self._numeric_cols()]

        for col in cols:
            data = self.df[col].dropna()

            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=("Violin / Box", "Scatter", "Histogram"),
                horizontal_spacing=0.08,
            )

            # Violin + Box
            fig.add_trace(
                go.Violin(
                    y=data,
                    box_visible=True,
                    meanline_visible=True,
                    name=col,
                    fillcolor="lightskyblue",
                    line_color="midnightblue",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

            # Scatter (index vs value)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode="markers",
                    marker=dict(size=4, opacity=0.6, color="teal"),
                    name="Index vs Value",
                ),
                row=1,
                col=2,
            )

            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=data,
                    marker_color="mediumpurple",
                    opacity=0.75,
                    name="Distribution",
                ),
                row=1,
                col=3,
            )

            fig.update_layout(
                title_text=f"Distribution Analysis: {col}",
                showlegend=False,
                height=400,
                template="plotly_white",
            )
            fig.show()

    def plot_relationship(self, col_x: str, col_y: str):
        """
        Detect column types and pick the right chart:
        - Num-Num → Scatter with OLS trendline
        - Cat-Num → Box plot with points
        - Cat-Cat → Grouped bar chart
        """
        if not self._check_data():
            return

        x_is_num = pd.api.types.is_numeric_dtype(self.df[col_x])
        y_is_num = pd.api.types.is_numeric_dtype(self.df[col_y])

        if x_is_num and y_is_num:
            fig = px.scatter(
                self.df,
                x=col_x,
                y=col_y,
                trendline="ols",
                title=f"{col_x} vs {col_y} (OLS Trendline)",
                template="plotly_white",
            )

        elif x_is_num != y_is_num:
            # One categorical, one numeric
            cat_col = col_x if not x_is_num else col_y
            num_col = col_y if not x_is_num else col_x
            fig = px.box(
                self.df,
                x=cat_col,
                y=num_col,
                points="all",
                title=f"{cat_col} vs {num_col}",
                template="plotly_white",
            )

        else:
            # Both categorical → grouped bar
            ct = pd.crosstab(self.df[col_x], self.df[col_y])
            fig = px.bar(
                ct,
                barmode="group",
                title=f"{col_x} vs {col_y} (Grouped Bar)",
                template="plotly_white",
            )

        fig.show()

    def plot_categorical(self, column_names: list | None = None):
        """
        Bar charts for each categorical column showing
        raw counts and percentage labels.
        """
        if not self._check_data():
            return
        cols = column_names or self._categorical_cols()

        for col in cols:
            if col not in self.df.columns:
                continue
            counts = self.df[col].value_counts()
            pct = (counts / counts.sum() * 100).round(1)
            labels = [f"{c} ({p}%)" for c, p in zip(counts.values, pct.values)]

            fig = go.Figure(
                go.Bar(
                    x=counts.index.astype(str),
                    y=counts.values,
                    text=labels,
                    textposition="outside",
                    marker_color="indianred",
                )
            )
            fig.update_layout(
                title=f"Category Frequency: {col}",
                xaxis_title=col,
                yaxis_title="Count",
                template="plotly_white",
                height=450,
            )
            fig.show()

    # ═══════════════ 5. Deep Statistical Insights ════════════════

    def plot_numerical_correlation(self):
        """Pearson correlation heatmap for all numeric columns."""
        if not self._check_data():
            return
        num_df = self.df[self._numeric_cols()]
        corr = num_df.corr(method="pearson")

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Heatmap (Numeric)",
            template="plotly_white",
            aspect="auto",
        )
        fig.show()

    def plot_categorical_correlation(self):
        """
        Cramér's V heatmap for all categorical columns.
        """
        if not self._check_data():
            return
        cat_cols = self._categorical_cols()
        if len(cat_cols) < 2:
            print("⚠  Need ≥ 2 categorical columns.")
            return

        n = len(cat_cols)
        matrix = pd.DataFrame(
            np.zeros((n, n)), index=cat_cols, columns=cat_cols
        )

        for i, c1 in enumerate(cat_cols):
            for j, c2 in enumerate(cat_cols):
                if i == j:
                    matrix.iloc[i, j] = 1.0
                elif j > i:
                    v = self._cramers_v(self.df[c1], self.df[c2])
                    matrix.iloc[i, j] = v
                    matrix.iloc[j, i] = v

        fig = px.imshow(
            matrix.astype(float),
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="Cramér's V Heatmap (Categorical)",
            template="plotly_white",
            aspect="auto",
        )
        fig.show()

    def plot_all_associations_heatmap(self):
        """
        Unified association heatmap across ALL columns:
        - Num-Num  → Pearson's r
        - Cat-Cat  → Cramér's V
        - Num-Cat  → Eta (correlation ratio via ANOVA)
        """
        if not self._check_data():
            return

        num_cols = self._numeric_cols()
        cat_cols = self._categorical_cols()
        all_cols = num_cols + cat_cols
        n = len(all_cols)
        matrix = pd.DataFrame(
            np.zeros((n, n)), index=all_cols, columns=all_cols
        )

        for i, c1 in enumerate(all_cols):
            for j, c2 in enumerate(all_cols):
                if i == j:
                    matrix.iloc[i, j] = 1.0
                elif j > i:
                    val = self._association(c1, c2, num_cols, cat_cols)
                    matrix.iloc[i, j] = val
                    matrix.iloc[j, i] = val

        fig = px.imshow(
            matrix.astype(float),
            text_auto=".2f",
            color_continuous_scale="Viridis",
            title="Unified Association Heatmap (All Types)",
            template="plotly_white",
            aspect="auto",
        )
        fig.show()

    # ── private statistics helpers ──

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Compute Cramér's V between two categorical Series."""
        ct = pd.crosstab(x, y)
        n = ct.sum().sum()
        if n == 0:
            return 0.0
        chi2 = stats.chi2_contingency(ct, correction=False)[0]
        r, k = ct.shape
        phi2 = chi2 / n
        # Bias correction
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        phi2_corr = max(0, phi2 - ((r - 1) * (k - 1)) / (n - 1))
        denom = min(r_corr - 1, k_corr - 1)
        if denom <= 0:
            return 0.0
        return np.sqrt(phi2_corr / denom)

    def _eta_correlation_ratio(
        self, categorical: pd.Series, numeric: pd.Series
    ) -> float:
        """Compute Eta (correlation ratio) via one-way ANOVA logic."""
        df_temp = pd.DataFrame({"cat": categorical, "num": numeric}).dropna()
        groups = [g["num"].values for _, g in df_temp.groupby("cat")]
        if len(groups) < 2:
            return 0.0
        grand_mean = df_temp["num"].mean()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2 for g in groups
        )
        ss_total = ((df_temp["num"] - grand_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        return np.sqrt(ss_between / ss_total)

    def _association(
        self,
        c1: str,
        c2: str,
        num_cols: list,
        cat_cols: list,
    ) -> float:
        """Pick the right association measure for a pair of columns."""
        c1_num = c1 in num_cols
        c2_num = c2 in num_cols

        if c1_num and c2_num:
            return self.df[[c1, c2]].corr().iloc[0, 1]
        elif not c1_num and not c2_num:
            return self._cramers_v(self.df[c1], self.df[c2])
        else:
            cat_col = c1 if not c1_num else c2
            num_col = c2 if not c1_num else c1
            return self._eta_correlation_ratio(
                self.df[cat_col], self.df[num_col]
            )


# ═══════════════════════════════════════════════════════════════════
#                        PlottingMethods
# ═══════════════════════════════════════════════════════════════════
class PlottingMethods:
    """
    Granular chart generation helper. Each method returns a
    dictionary ``{'status': 'ok', 'html': '<plotly html>'}``
    for flexible embedding in Colab.
    """

    def __init__(self):
        self.df: pd.DataFrame | None = None

    @staticmethod
    def _fig_to_result(fig) -> dict:
        """Convert a Plotly figure to an HTML result dict."""
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        return {"status": "ok", "html": html}

    @staticmethod
    def display_image(result: dict):
        """
        Render an HTML result dict inside a Colab / Jupyter notebook.
        """
        if result is None or result.get("status") != "ok":
            print("⚠  Nothing to display.")
            return
        try:
            from IPython.display import display, HTML
            display(HTML(result["html"]))
        except ImportError:
            print("Display requires IPython / Jupyter.")

    def get_methods_info(self) -> dict:
        """Return a summary table of available plotting methods."""
        info = [
            {
                "method": "plot_bar_chart",
                "description": "Bar chart (grouped / stacked)",
                "required_params": "x, y, data",
            },
            {
                "method": "plot_pie_chart",
                "description": "Pie / donut chart",
                "required_params": "names, values, data",
            },
            {
                "method": "plot_histogram",
                "description": "Histogram with optional bins",
                "required_params": "x, data",
            },
            {
                "method": "plot_heat_map",
                "description": "Pivot heatmap with aggregation",
                "required_params": "values, index, columns, data",
            },
            {
                "method": "plot_sankey_diagram",
                "description": "Sankey flow diagram",
                "required_params": "source_column, target_column, values, data",
            },
            {
                "method": "plot_simple_sunburst_graph",
                "description": "Sunburst hierarchical chart",
                "required_params": "path, values, data",
            },
        ]
        return {"status": "ok", "response": info}

    # ─────────── Chart Methods ───────────

    def plot_bar_chart(
        self,
        x: str,
        y: str,
        data: pd.DataFrame | None = None,
        color: str | None = None,
        barmode: str = "group",
        title: str | None = None,
    ) -> dict:
        """
        Create a grouped or stacked bar chart.

        Parameters
        ----------
        x, y : str   Column names.
        data : DataFrame
        color : str, optional   Column for colour grouping.
        barmode : str   ``'group'`` or ``'stack'``.
        title : str, optional
        """
        df = data if data is not None else self.df
        if df is None:
            return {"status": "error", "html": ""}
        title = title or f"{y} by {x}"
        fig = px.bar(
            df, x=x, y=y, color=color, barmode=barmode,
            title=title, template="plotly_white",
        )
        return self._fig_to_result(fig)

    def plot_pie_chart(
        self,
        names: str,
        values: str,
        data: pd.DataFrame | None = None,
        hole: float = 0.0,
        title: str | None = None,
    ) -> dict:
        """
        Pie or donut chart.

        Parameters
        ----------
        names : str   Category column.
        values : str   Numeric column for sizing.
        hole : float   0 for pie, 0.3–0.5 for donut.
        """
        df = data if data is not None else self.df
        if df is None:
            return {"status": "error", "html": ""}
        title = title or f"{values} by {names}"
        fig = px.pie(
            df, names=names, values=values, hole=hole,
            title=title, template="plotly_white",
        )
        return self._fig_to_result(fig)

    def plot_histogram(
        self,
        x: str,
        data: pd.DataFrame | None = None,
        bins: list | None = None,
        title: str | None = None,
    ) -> dict:
        """
        Histogram with optional custom bin edges.
        """
        df = data if data is not None else self.df
        if df is None:
            return {"status": "error", "html": ""}
        title = title or f"Distribution of {x}"
        if bins:
            fig = px.histogram(
                df, x=x, title=title,
                template="plotly_white",
                nbins=len(bins),
            )
        else:
            fig = px.histogram(
                df, x=x, title=title, template="plotly_white",
            )
        return self._fig_to_result(fig)

    def plot_heat_map(
        self,
        values: str,
        index: str,
        columns: str,
        data: pd.DataFrame | None = None,
        aggregade_method: str = "mean",
        title: str | None = None,
    ) -> dict:
        """
        Pivot-table based heatmap with a chosen aggregation.
        """
        df = data if data is not None else self.df
        if df is None:
            return {"status": "error", "html": ""}
        pivot = pd.pivot_table(
            df, values=values, index=index, columns=columns,
            aggfunc=aggregade_method,
        )
        title = title or f"{values} ({aggregade_method})"
        fig = px.imshow(
            pivot,
            text_auto=".2f",
            title=title,
            template="plotly_white",
            aspect="auto",
            color_continuous_scale="YlGnBu",
        )
        return self._fig_to_result(fig)

    def plot_sankey_diagram(
        self,
        source_column: str,
        target_column: str,
        values: str,
        data: pd.DataFrame | None = None,
        title: str | None = None,
    ) -> dict:
        """
        Sankey flow diagram between two categorical columns.
        """
        df = data if data is not None else self.df
        if df is None:
            return {"status": "error", "html": ""}

        grouped = df.groupby([source_column, target_column])[values].sum().reset_index()

        all_labels = list(
            pd.concat([grouped[source_column], grouped[target_column]])
            .astype(str)
            .unique()
        )
        label_map = {label: i for i, label in enumerate(all_labels)}

        fig = go.Figure(
            go.Sankey(
                node=dict(label=all_labels, pad=15, thickness=20),
                link=dict(
                    source=[label_map[str(s)] for s in grouped[source_column]],
                    target=[label_map[str(t)] for t in grouped[target_column]],
                    value=grouped[values].tolist(),
                ),
            )
        )
        fig.update_layout(
            title_text=title or f"{source_column} → {target_column}",
            template="plotly_white",
        )
        return self._fig_to_result(fig)

    def plot_simple_sunburst_graph(
        self,
        path: list,
        values: str,
        data: pd.DataFrame | None = None,
        title: str | None = None,
    ) -> dict:
        """
        Sunburst hierarchical chart.

        Parameters
        ----------
        path : list   List of column names defining the hierarchy.
        values : str  Numeric column for segment size.
        """
        df = data if data is not None else self.df
        if df is None:
            return {"status": "error", "html": ""}
        fig = px.sunburst(
            df, path=path, values=values,
            title=title or "Sunburst",
            template="plotly_white",
        )
        return self._fig_to_result(fig)
