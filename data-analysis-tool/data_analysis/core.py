import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class DataInspector:
    """End-to-end data sanitization, feature engineering, and exploration engine."""
    
    def __init__(self, df=None):
        self.df = df

    def upload_data(self):
        """Handles local file uploads in Google Colab."""
        try:
            from google.colab import files
            uploaded = files.upload()
            if not uploaded: return
            file_name = list(uploaded.keys())[0]
            self.df = pd.read_csv(io.BytesIO(uploaded[file_name]))
            self._initial_cleanup()
            print(f"Successfully loaded {file_name}")
        except ImportError:
            print("Upload feature is only available in Google Colab.")

    def _initial_cleanup(self):
        """Handles garbage strings and type correction."""
        garbage_strings = ['?', 'n/a', 'NULL', ' ', 'N/A', 'null']
        if self.df is not None:
            self.df.replace(garbage_strings, np.nan, inplace=True)
            for col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')

    def display_summary(self):
        """Displays row/column counts, a 20-row preview, and type breakdowns."""
        if self.df is None: 
            print("No data loaded.")
            return
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        print(f"Dataset Shape: {self.df.shape[0]} Rows, {self.df.shape[1]} Columns")
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        print("\nFirst 20 Rows Preview:")
        from IPython.display import display
        display(self.df.head(20))

    def handle_missing_values(self, strategy='median', fill_value=None):
        """Imputes missing values (mean, median, mode, constant)."""
        if self.df is None: return
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    val = self.df[col].mean() if strategy == 'mean' else self.df[col].median()
                    self.df[col] = self.df[col].fillna(val)
                else:
                    val = self.df[col].mode()[0] if (strategy == 'mode' and not self.df[col].mode().empty) else fill_value
                    if val is not None:
                        self.df[col] = self.df[col].fillna(val)
        print(f"Missing values handled using '{strategy}' strategy.")

    def remove_duplicates(self):
        """Prunes exact row matches."""
        if self.df is None: return
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_len - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns=None, delete=False):
        """IQR-based outlier detection."""
        if self.df is None: return
        cols = columns if columns else self.df.select_dtypes(include=[np.number]).columns
        for col in cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (self.df[col] < Q1 - 1.5 * IQR) | (self.df[col] > Q3 + 1.5 * IQR)
            if delete: self.df = self.df[~outliers]
            else: print(f"Identified {outliers.sum()} outliers in {col}.")

    def delete_rows(self, indices_str):
        """Deletes rows based on comma-separated string."""
        if self.df is None: return
        indices = [int(i.strip()) for i in indices_str.split(',') if i.strip().isdigit()]
        self.df.drop(index=indices, inplace=True, errors='ignore')

    def delete_columns(self, cols_str):
        """Deletes columns based on comma-separated string."""
        if self.df is None: return
        cols = [c.strip() for c in cols_str.split(',')]
        self.df.drop(columns=cols, inplace=True, errors='ignore')

    def extract_normalized_numeric_data(self, method='standard', exclude_cols=None):
        """Scales numeric data."""
        if self.df is None: return None
        exclude = exclude_cols or []
        num_cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in exclude]
        scaler = {'minmax': MinMaxScaler(), 'standard': StandardScaler(), 'robust': RobustScaler()}[method]
        df_norm = self.df.copy()
        df_norm[num_cols] = scaler.fit_transform(df_norm[num_cols])
        return df_norm

    def extract_normalized_categorical_data(self):
        """Encodes categorical data."""
        if self.df is None: return None
        return pd.get_dummies(self.df.select_dtypes(exclude=[np.number]), drop_first=True)

    def create_normalized_data_df(self):
        """Unified dataframe merger."""
        if self.df is None: return None
        return pd.concat([self.extract_normalized_numeric_data(), self.extract_normalized_categorical_data()], axis=1)

    def plot_relationship(self, col1, col2):
        """Detects types and chooses chart: Scatter, Box, or Bar."""
        if self.df is None: return
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
        if is_num1 and is_num2: fig = px.scatter(self.df, x=col1, y=col2, trendline="ols")
        elif not is_num1 and not is_num2: fig = px.bar(self.df.groupby([col1, col2]).size().reset_index(name='Count'), x=col1, y='Count', color=col2, barmode='group')
        else: fig = px.box(self.df, x=col1 if not is_num1 else col2, y=col2 if not is_num1 else col1, points="all")
        fig.show()

    def plot_all_associations_heatmap(self):
        """Unified Pearson (Num) and Cramer's V (Cat) heatmap."""
        if self.df is None: return
        cols = self.df.columns
        corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i in cols:
            for j in cols:
                if i == j: corr.loc[i, j] = 1.0
                elif pd.api.types.is_numeric_dtype(self.df[i]) and pd.api.types.is_numeric_dtype(self.df[j]):
                    corr.loc[i, j] = self.df[i].corr(self.df[j])
                else:
                    crosstab = pd.crosstab(self.df[i], self.df[j])
                    chi2 = stats.chi2_contingency(crosstab)[0]
                    n = crosstab.sum().sum()
                    phi2 = chi2 / n
                    corr.loc[i, j] = np.sqrt(phi2 / min(crosstab.shape) - 1) if min(crosstab.shape) > 1 else 0
        fig = px.imshow(corr, text_auto=".2f", title="Unified Association Heatmap")
        fig.show()
