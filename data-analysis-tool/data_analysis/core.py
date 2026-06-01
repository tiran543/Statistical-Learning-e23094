import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as ss
from IPython.display import display, HTML
import io

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

class PlottingMethods:
    """Handles granular chart generation returning HTML-wrapped figures."""
    
    @staticmethod
    def display_image(fig):
        """Displays a Plotly figure in Colab."""
        if fig is not None:
            fig.show()
            
    def plot_bar_chart(self, data, x, y, color=None, barmode='group'):
        """Creates a stacked or grouped bar chart."""
        if data is None or data.empty:
            return None
        fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, title=f"Bar Chart: {y} by {x}")
        return fig

    def plot_pie_chart(self, data, names, values, hole=0.4, title="Pie Chart"):
        """Generates a responsive pie chart with a donut hole."""
        if data is None or data.empty:
            return None
        fig = px.pie(data, names=names, values=values, hole=hole, title=title)
        return fig

    def plot_histogram(self, data, x, bins=None, title="Histogram"):
        """Plots distribution with custom bin intervals."""
        if data is None or data.empty:
            return None
        fig = px.histogram(data, x=x, nbins=len(bins)-1 if bins else None, title=title)
        return fig


class DataInspector:
    """End-to-end tool for data ingestion, cleaning, and exploration."""
    
    def _init_(self, df=None):
        self.df = df
        self.plotter = PlottingMethods()
        if self.df is not None:
            self._initial_cleanup()

    def upload_data(self):
        """Handles local file uploads in Google Colab."""
        if not IN_COLAB:
            print("Upload feature is only available in Google Colab.")
            return
        
        print("Please upload your CSV file:")
        uploaded = files.upload()
        for filename in uploaded.keys():
            self.df = pd.read_csv(io.BytesIO(uploaded[filename]))
            print(f"Successfully loaded {filename}")
            self._initial_cleanup()
            break

    def _initial_cleanup(self):
        """Internal method to handle garbage strings and type correction."""
        garbage_strings = ['?', 'n/a', 'NULL', ' ', 'N/A', 'null']
        self.df.replace(garbage_strings, np.nan, inplace=True)
        
        for col in self.df.columns:
            try:
                numeric_col = pd.to_numeric(self.df[col])
                if not numeric_col.isna().all():
                    self.df[col] = numeric_col
            except ValueError:
                continue

    def get_summary(self):
        """Displays row/col counts, a 20-row preview, and num/cat breakdown."""
        if self.df is None: return "No data loaded."
        
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        
        print(f"Dataset Shape: {self.df.shape[0]} Rows, {self.df.shape[1]} Columns")
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        print("\nFirst 20 Rows Preview:")
        display(self.df.head(20))

    def handle_missing_values(self, strategy='median', fill_value=None):
        """Imputes missing values using mean, median, mode, or constant."""
        if self.df is None: return
        
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                if self.df[col].dtype in [np.float64, np.int64]:
                    if strategy == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif strategy == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif strategy == 'constant' and fill_value is not None:
                        self.df[col].fillna(fill_value, inplace=True)
                else:
                    if strategy == 'mode':
                        self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    elif strategy == 'constant' and fill_value is not None:
                        self.df[col].fillna(fill_value, inplace=True)
        print(f"Missing values handled using '{strategy}' strategy.")

    def remove_duplicates(self):
        """Prunes exact row matches."""
        if self.df is None: return
        initial_rows = len(self.df)
        self.df.drop_duplicates(inplace=True)
        print(f"Removed {initial_rows - len(self.df)} duplicate rows.")

    def handle_outliers(self, columns=None, delete=False):
        """IQR-based outlier detection system."""
        if self.df is None: return
        cols = columns if columns else self.df.select_dtypes(include=np.number).columns
        
        for col in cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            if delete:
                self.df = self.df[~outliers]
            else:
                print(f"Identified {outliers.sum()} outliers in {col} (Bounds: {lower_bound} to {upper_bound}).")

    def delete_columns(self):
        """Interactive column deletion."""
        cols_to_drop = input("Enter columns to delete (comma-separated): ").split(',')
        cols_to_drop = [c.strip() for c in cols_to_drop if c.strip() in self.df.columns]
        self.df.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped columns: {cols_to_drop}")

    def extract_normalized_numeric_data(self, method='standard'):
        """Scales numeric data using minmax, standard, or robust scaling."""
        if self.df is None: return None
        num_df = self.df.select_dtypes(include=np.number).copy()
        
        for col in num_df.columns:
            if method == 'standard': 
                num_df[col] = (num_df[col] - num_df[col].mean()) / num_df[col].std()
            elif method == 'minmax':
                num_df[col] = (num_df[col] - num_df[col].min()) / (num_df[col].max() - num_df[col].min())
            elif method == 'robust': 
                Q1, Q3 = num_df[col].quantile(0.25), num_df[col].quantile(0.75)
                num_df[col] = (num_df[col] - num_df[col].median()) / (Q3 - Q1)
        return num_df

    def extract_normalized_categorical_data(self, method='onehot'):
        """Encodes categorical data using onehot, ordinal, or uniform."""
        if self.df is None: return None
        cat_df = self.df.select_dtypes(exclude=np.number).copy()
        
        if method == 'onehot':
            return pd.get_dummies(cat_df, drop_first=True)
        elif method == 'ordinal':
            for col in cat_df.columns:
                cat_df[col] = cat_df[col].astype('category').cat.codes
            return cat_df
        return cat_df

    def create_normalized_data_df(self):
        """Merges scaled numeric and encoded categorical data."""
        num_df = self.extract_normalized_numeric_data(method='standard')
        cat_df = self.extract_normalized_categorical_data(method='onehot')
        return pd.concat([num_df, cat_df], axis=1)

    def plot_numerical(self, columns=None):
        """Generates a 3-panel subplot for numeric columns."""
        cols = columns if columns else self.df.select_dtypes(include=np.number).columns
        for col in cols:
            fig = make_subplots(rows=1, cols=3, subplot_titles=("Box Plot", "Scatter Plot", "Histogram"))
            fig.add_trace(go.Box(x=self.df[col], name=col), row=1, col=1)
            fig.add_trace(go.Scatter(y=self.df[col], mode='markers', name=col), row=1, col=2)
            fig.add_trace(go.Histogram(x=self.df[col], name=col), row=1, col=3)
            fig.update_layout(title_text=f"Univariate Analysis: {col}", showlegend=False)
            fig.show()

    def plot_relationship(self, col1, col2):
        """Smart relationship plotting based on data types."""
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
        
        if is_num1 and is_num2: 
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Scatter: {col1} vs {col2}")
        elif not is_num1 and not is_num2: 
            count_df = self.df.groupby([col1, col2]).size().reset_index(name='Count')
            fig = px.bar(count_df, x=col1, y="Count", color=col2, barmode='group', title=f"Grouped Bar: {col1} vs {col2}")
        else: 
            cat_col, num_col = (col1, col2) if not is_num1 else (col2, col1)
            fig = px.box(self.df, x=cat_col, y=num_col, points="all", title=f"Boxplot: {num_col} grouped by {cat_col}")
        fig.show()

    def plot_all_associations_heatmap(self):
        """Generates a unified heatmap visualizing relationships."""
        num_df = self.df.select_dtypes(include=np.number)
        if not num_df.empty:
            corr_matrix = num_df.corr(method='pearson')
            fig = px.imshow(corr_matrix, text_auto=True, title="Pearson Correlation Heatmap (Numeric)")
            fig.show()
        else:
            print("No numeric columns available for standard Pearson correlation.")
