import dash
from dash import dash_table, html
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45]
})

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in df.columns],
        page_size=2  # Number of rows per page
    )
])

if __name__ == '__main__':
    app.run(debug=True)
